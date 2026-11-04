--[[
tex2docx.lua - pandoc Lua filter used by build_docx.py to turn report.tex into an
editable Word document. It runs AFTER --citeproc, so it also sees the reference list.

Each rule reproduces what LaTeX prints for the same source:
  \maketitle         -> title page (logo, title lines, author, date), keeping the \vspace gaps
  \newpage           -> Word page break
  \tableofcontents   -> "Contents" heading + Word TOC field (levels 1-3, article's tocdepth)
  \ref{label}        -> the table's number (tables are numbered in document order, as in LaTeX)
  \section* etc.     -> heading styled like a section but kept out of the TOC
  \paragraph{...}    -> bold run-in lead-in at the start of the paragraph
  p{3.2cm} columns   -> proportional Word column widths
  \footnotesize table-> 10 pt cell text (\small tables use the 11 pt "Compact" style)
  $\rightarrow$ etc. -> the plain Unicode symbol (anything more complex stays a Word equation)
  "et al.\ "         -> ordinary space (pandoc reads "\ " as a no-break space)
  enumerate/itemize  -> "List Paragraph" style, spaced like the body text
Every raw LaTeX command that is dropped is reported on stderr so nothing disappears silently.
]]

local function warn(msg) io.stderr:write('[tex2docx] ' .. msg .. '\n') end

local function is_tex(fmt) return fmt == 'latex' or fmt == 'tex' end

-- ------------------------------------------------------------------ helpers
local TWIPS = { cm = 566.93, mm = 56.693, ['in'] = 1440, pt = 20, bp = 20.075, em = 240, ex = 103 }

local function to_twips(len)
  local num, unit = len:match('^%s*([%d%.]+)%s*(%a%a)%s*$')
  if not num or not TWIPS[unit] then return nil end
  return math.floor(tonumber(num) * TWIPS[unit] + 0.5)
end

-- empty paragraph of an exact height: reproduces a \vspace gap
local function spacer(twips)
  return pandoc.RawBlock('openxml', ('<w:p><w:pPr><w:spacing w:before="0" w:after="0" '
    .. 'w:line="%d" w:lineRule="exact"/></w:pPr></w:p>'):format(twips))
end

-- page break in a 1 pt-high paragraph, so it adds no visible blank line
local PAGE_BREAK = pandoc.RawBlock('openxml', '<w:p><w:pPr><w:spacing w:before="0" w:after="0" '
  .. 'w:line="20" w:lineRule="exact"/></w:pPr><w:r><w:br w:type="page"/></w:r></w:p>')

local function toc_block(title)
  return pandoc.RawBlock('openxml', table.concat({
    '<w:sdt><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents"/>',
    '<w:docPartUnique/></w:docPartObj></w:sdtPr><w:sdtContent>',
    '<w:p><w:pPr><w:pStyle w:val="TOCHeading"/></w:pPr><w:r><w:t>', title, '</w:t></w:r></w:p>',
    '<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r>',
    '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>',
    '<w:r><w:fldChar w:fldCharType="separate"/></w:r>',
    '<w:r><w:t>Right-click here and choose Update Field to build the table of contents.</w:t></w:r>',
    '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>',
    '</w:sdtContent></w:sdt>' }))
end

local function styled(style, inlines)
  return pandoc.Div({ pandoc.Para(inlines) }, { ['custom-style'] = style })
end

local function is_space(el) return el.t == 'Space' or el.t == 'SoftBreak' end

local function trim(inls)
  while #inls > 0 and is_space(inls[1]) do inls:remove(1) end
  while #inls > 0 and is_space(inls[#inls]) do inls:remove(#inls) end
  return inls
end

local function as_list(v)
  if v == nil then return {} end
  if pandoc.utils.type(v) == 'List' then return v end
  return { v }
end

-- ------------------------------------------------------------------ title page
-- "\\ \vspace{X}" inside \title starts a new paragraph after an X-high gap;
-- a bare "\\" stays a line break inside the same paragraph.
local function title_page(meta)
  local blocks, cur = pandoc.List(), pandoc.List()
  local function flush()
    cur = trim(cur)
    if #cur > 0 then blocks:insert(styled('Title', cur)) end
    cur = pandoc.List()
  end
  local title = meta.title or pandoc.Inlines({})
  local i = 1
  while i <= #title do
    local el = title[i]
    if el.t == 'LineBreak' then
      local j = i + 1
      while j <= #title and is_space(title[j]) do j = j + 1 end
      local nxt = title[j]
      local gap = nxt and nxt.t == 'RawInline' and nxt.text:match('^\\vspace%*?%s*{(.-)}%s*$')
      if gap then
        flush()
        local tw = to_twips(gap)
        if tw then blocks:insert(spacer(tw)) else warn('title: unknown \\vspace length ' .. gap) end
        i = j + 1
      else
        cur:insert(el)
        i = j
      end
    elseif el.t == 'RawInline' then
      warn('title: dropped ' .. el.text)
      i = i + 1
    else
      cur:insert(el)
      i = i + 1
    end
  end
  flush()

  local names = pandoc.Inlines({})
  for k, a in ipairs(as_list(meta.author)) do
    if k > 1 then names:insert(pandoc.LineBreak()) end
    names:extend(a)
  end
  if #names > 0 then blocks:insert(styled('Author', names)) end
  if meta.date then blocks:insert(styled('Date', meta.date)) end
  return blocks
end

-- ------------------------------------------------------------------ tables
local function read_source()
  local f = PANDOC_STATE.input_files[1]
  local h = f and io.open(f, 'rb')
  if not h then return '' end
  local s = h:read('a')
  h:close()
  s = s:gsub('^%%[^\n]*', ''):gsub('\n%%[^\n]*', '\n'):gsub('([^\\])%%[^\n]*', '%1')
  return s
end

local function parse_colspec(spec)
  local cols, i = {}, 1
  while i <= #spec do
    local ch = spec:sub(i, i)
    if ch == 'l' or ch == 'c' or ch == 'r' then
      cols[#cols + 1] = {}
      i = i + 1
    elseif ch == 'p' or ch == 'm' or ch == 'b' then
      local s, e = spec:find('%b{}', i + 1)
      cols[#cols + 1] = { width = s and to_twips(spec:sub(s + 1, e - 1)) }
      i = (e or i) + 1
    elseif ch == '@' or ch == '!' or ch == '>' or ch == '<' then
      local _, e = spec:find('%b{}', i + 1)
      i = (e or i) + 1
    else
      i = i + 1 -- '|', spaces
    end
  end
  return cols
end

-- column specs of every \begin{tabular}{...}, in source order
local function tabular_specs(src)
  local out, pos = {}, 1
  while true do
    local s, e = src:find('\\begin{tabular}', pos, true)
    if not s then break end
    local rest = e + 1
    local _, oe = src:find('^%s*%b[]', rest)
    if oe then rest = oe + 1 end
    local bs, be = src:find('^%s*%b{}', rest)
    if bs then
      out[#out + 1] = parse_colspec(src:sub(bs, be):match('{(.*)}'))
      pos = be + 1
    else
      pos = e + 1
    end
  end
  return out
end

local function find_table(b)
  if b.t == 'Table' then return b end
  if b.t == 'Div' then
    for _, c in ipairs(b.content) do
      if c.t == 'Table' then return c end
    end
  end
  return nil
end

local SMALLER = { footnotesize = true, scriptsize = true, tiny = true }

local function fix_table(tbl, spec, size)
  -- widths: only when every column is a fixed-width p{}/m{}/b{} column
  if spec and #spec == #tbl.colspecs then
    local total, all = 0, true
    for _, c in ipairs(spec) do
      if c.width then total = total + c.width else all = false end
    end
    if all and total > 0 then
      local cs = pandoc.List()
      for k, c in ipairs(tbl.colspecs) do cs:insert({ c[1], spec[k].width / total }) end
      tbl.colspecs = cs
    end
  elseif spec then
    warn(('table: %d columns in source spec, %d in pandoc table; widths left automatic')
      :format(#spec, #tbl.colspecs))
  end
  -- \footnotesize (or smaller) tables get 10 pt cell text; the caption is left alone
  if size and SMALLER[size] then
    local cap = tbl.caption
    tbl = tbl:walk({
      Plain = function(p)
        return pandoc.Plain({ pandoc.Span(p.content, { ['custom-style'] = 'Table Text Small' }) })
      end,
    })
    tbl.caption = cap
  end
  return tbl
end

local function find_figure(b)
  if b.t == 'Figure' then return b end
  if b.t == 'Div' then
    for _, c in ipairs(b.content) do
      if c.t == 'Figure' then return c end
    end
  end
  return nil
end

local function replace_table(b, tbl)
  if b.t == 'Table' then return tbl end
  local content = pandoc.List()
  for _, c in ipairs(b.content) do content:insert(c.t == 'Table' and tbl or c) end
  b.content = content
  return b
end

-- ------------------------------------------------------------------ inline fixes
local SYMBOL = {
  ['\\rightarrow'] = '→', ['\\to'] = '→', ['\\leftarrow'] = '←', ['\\Rightarrow'] = '⇒',
  ['\\approx'] = '≈', ['\\sim'] = '∼', ['\\times'] = '×', ['\\pm'] = '±',
  ['\\geq'] = '≥', ['\\ge'] = '≥', ['\\leq'] = '≤', ['\\le'] = '≤', ['>'] = '>', ['<'] = '<',
}

local NBSP = '\u{a0}'
local PAGEBREAK = { newpage = true, clearpage = true, cleardoublepage = true, pagebreak = true }
local SIZE = { small = true, footnotesize = true, scriptsize = true, tiny = true,
               normalsize = true, large = true }

-- ------------------------------------------------------------------ main
function Pandoc(doc)
  local specs = tabular_specs(read_source())
  local blocks = doc.blocks

  -- pass 1: numbers of sections and captioned tables in document order.
  local number, n, ntables, figure_n = {}, 0, 0, 0
  local section = { 0, 0, 0 }
  for _, b in ipairs(blocks) do
    if b.t == 'Header' and b.level <= 3 and not b.classes:includes('unnumbered') then
      section[b.level] = section[b.level] + 1
      for level = b.level + 1, 3 do section[level] = 0 end
      if b.identifier and b.identifier ~= '' then
        local parts = {}
        for level = 1, b.level do parts[#parts + 1] = tostring(section[level]) end
        number[b.identifier] = table.concat(parts, '.')
      end
    end
    local tb = find_table(b)
    if tb then
      ntables = ntables + 1
      if #tb.caption.long > 0 then
        n = n + 1
        if b.identifier and b.identifier ~= '' then number[b.identifier] = n end
      end
    end
    local fig = find_figure(b)
    if fig then
      figure_n = figure_n + 1
      local identifier = (fig.identifier and fig.identifier ~= '') and fig.identifier or b.identifier
      if identifier and identifier ~= '' then number[identifier] = figure_n end
    end
  end
  if #specs ~= ntables then
    warn(('found %d tabular specs but %d tables; column widths left automatic'):format(#specs, ntables))
    specs = {}
  end

  -- pass 2: rebuild the block list
  local out, size, t, i = pandoc.List(), nil, 0, 1
  while i <= #blocks do
    local b, nxt = blocks[i], blocks[i + 1]
    local tbl = find_table(b)
    if b.t == 'RawBlock' and is_tex(b.format) then
      local cmd = b.text:match('^\\(%a+)')
      if cmd == 'maketitle' then
        out:extend(title_page(doc.meta))
      elseif PAGEBREAK[cmd] then
        out:insert(PAGE_BREAK)
      elseif cmd == 'tableofcontents' then
        out:insert(toc_block('Contents'))
      elseif SIZE[cmd] then
        size = cmd
      elseif cmd ~= 'centering' and cmd ~= 'linespread' and cmd ~= 'bibliographystyle' then
        warn('dropped raw block ' .. b.text)
      end
      i = i + 1
    elseif tbl then
      t = t + 1
      out:insert(replace_table(b, fix_table(tbl, specs[t], size)))
      size = nil
      i = i + 1
    elseif b.t == 'Header' and b.level <= 3 and b.classes:includes('unnumbered') then
      out:insert(styled('Unnumbered Heading ' .. b.level, b.content))
      if nxt and nxt.t == 'Para' then -- LaTeX does not indent the first paragraph after a heading
        out:insert(styled('First Paragraph', nxt.content))
        i = i + 1
      end
      i = i + 1
    elseif b.t == 'OrderedList' or b.t == 'BulletList' then
      -- list items otherwise fall back to Word's single-spaced "Normal" style
      out:insert(pandoc.Div({ b }, { ['custom-style'] = 'List Paragraph' }))
      i = i + 1
    elseif b.t == 'Header' and b.level >= 4 then
      local lead = pandoc.Inlines({ pandoc.Strong(b.content) })
      if nxt and nxt.t == 'Para' then
        lead:insert(pandoc.Space())
        lead:extend(nxt.content)
        i = i + 1
      end
      out:insert(styled('Run-in Paragraph', lead))
      i = i + 1
    else
      out:insert(b)
      i = i + 1
    end
  end

  local meta = doc.meta
  meta.title, meta.author, meta.date = nil, nil, nil -- rendered as the title page above
  meta.bibliography, meta['reference-section-title'] = nil, nil -- already used by citeproc

  return pandoc.Pandoc(out, meta):walk({
    RawInline = function(r)
      if not is_tex(r.format) then return nil end
      local cmd, arg = r.text:match('^\\(%a+)%s*{(.-)}%s*$')
      if cmd == 'ref' then
        if number[arg] then return pandoc.Str(tostring(number[arg])) end
        warn('unresolved \\ref{' .. arg .. '}')
        return pandoc.Str('??')
      end
      warn('dropped raw inline ' .. r.text)
      return {}
    end,
    Math = function(m)
      if m.mathtype == 'InlineMath' then
        local sym = SYMBOL[(m.text:gsub('^%s+', ''):gsub('%s+$', ''))]
        if sym then return pandoc.Str(sym) end
      end
    end,
    -- pandoc reads "et al.\ " as a no-break space; in LaTeX "\ " is an ordinary breakable space
    Str = function(s)
      if not s.text:find(NBSP, 1, true) then return nil end
      local parts = pandoc.List()
      for piece in (s.text .. NBSP):gmatch('(.-)' .. NBSP) do
        if #parts > 0 then parts:insert(pandoc.Space()) end
        if piece ~= '' then parts:insert(pandoc.Str(piece)) end
      end
      return parts
    end,
  })
end
