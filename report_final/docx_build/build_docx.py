#!/usr/bin/env python3
"""Build report.docx, an editable Word version of report.tex.

    python build_docx.py               # writes ../report.docx
    python build_docx.py --no-word     # skip the Microsoft Word step
    python build_docx.py --pdf X.pdf   # also export a PDF through Word, to check the layout

Steps
1. reference.docx: pandoc's default Word template restyled to look like the LaTeX
   article (A4, 12 pt Times New Roman, 1.5 line spacing, indented paragraphs, black bold
   headings, booktabs-style tables, centred page numbers).
2. pandoc reads report.tex, numbers the citations in IEEE style (ieee.csl, like ieeetr)
   and runs tex2docx.lua (title page, page breaks, contents, table numbers and widths,
   run-in paragraph headings).
3. Each table is kept on one page together with its caption (LaTeX tables never split).
4. If Microsoft Word is installed it opens the file once to fill in the table of
   contents with page numbers, then saves it.

Needs pandoc (pip install pypandoc_binary), python-docx, and pywin32 for step 4.
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE.parent
TEX = REPORT_DIR / "report.tex"
OUT = REPORT_DIR / "report.docx"
REFERENCE = HERE / "reference.docx"
LUA_FILTER = HERE / "tex2docx.lua"
CSL = HERE / "ieee.csl"

FONT = "Times New Roman"
LINE_1_5 = 360  # Word "1.5 lines": the setspace \onehalfspacing that \linespread{1.241} gives at 12 pt


# --------------------------------------------------------------------------- pandoc
def find_pandoc() -> str:
    exe = shutil.which("pandoc")
    if exe:
        return exe
    try:
        import pypandoc

        return pypandoc.get_pandoc_path()
    except (ImportError, OSError):
        sys.exit("pandoc not found. Install it with:  pip install pypandoc_binary")


def bibliography_file() -> Path:
    m = re.search(r"\\bibliography\{([^}]+)\}", TEX.read_text(encoding="utf-8"))
    if not m:
        sys.exit(f"no \\bibliography{{...}} found in {TEX.name}")
    name = m.group(1).split(",")[0].strip()
    return REPORT_DIR / (name if name.endswith(".bib") else name + ".bib")


def _closing_brace(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def protect_case(bib: str) -> str:
    """Make pandoc print titles the way ieeetr.bst does. pandoc sentence-cases every title-like
    field; ieeetr prints booktitle/journal exactly as typed ("(S&P)", not "(s&p)") and keeps the
    case of a title's first character ("gpt-oss", not "Gpt-oss"). Braces protect text from
    pandoc's case changes. Only a temporary copy is changed, never myBibFile.bib."""
    out, pos = [], 0
    for m in re.finditer(r"\b(booktitle|journal|title)\s*=\s*\{", bib, re.IGNORECASE):
        start = m.end() - 1
        end = _closing_brace(bib, start)
        if start < pos or end < 0:
            continue
        value = bib[start + 1 : end]
        if m.group(1).lower() == "title":
            first = re.match(r"(\s*)([a-z][^\s{}]*)", value)
            if not first:
                continue
            value = f"{first.group(1)}{{{first.group(2)}}}{value[first.end():]}"
        else:
            value = "{" + value + "}"
        out += [bib[pos : start + 1], value, "}"]
        pos = end + 1
    out.append(bib[pos:])
    return "".join(out)


def expand_tex_inputs(path: Path, *, seen: set[Path] | None = None) -> str:
    """Expand local ``\\input{...}`` files for Pandoc's LaTeX reader.

    The PDF build can let LaTeX resolve inputs directly. Pandoc does not reliably
    expand them, so the editable-DOCX path uses this temporary, recursive expansion.
    """
    resolved = path.resolve()
    visited = set() if seen is None else seen
    if resolved in visited:
        raise RuntimeError(f"recursive TeX input: {resolved}")
    visited.add(resolved)
    text = resolved.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        include = Path(match.group(1))
        if include.suffix == "":
            include = include.with_suffix(".tex")
        target = (resolved.parent / include).resolve()
        if not target.exists():
            raise FileNotFoundError(f"TeX input not found: {target}")
        return expand_tex_inputs(target, seen=set(visited))

    return re.sub(r"\\input\{([^}]+)\}", replace, text)


def prepare_pandoc_tex(path: Path) -> str:
    """Expand inputs, drop LaTeX-only table-resize wrappers, and remove the page break
    between the contents and the first section. The DOCX build starts the body on a new
    page with a paragraph property (see finish()) instead of an explicit break, which
    avoids a stray blank page when the contents happen to fill their page."""
    text = expand_tex_inputs(path)
    text = re.sub(
        r"\\resizebox\{\\textwidth\}\{!\}\{%\s*(\\begin\{tabular\}.*?\\end\{tabular\})\s*\}",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    return re.sub(r"(\\tableofcontents\s*)\\newpage", r"\1", text, count=1)


def run_pandoc(exe: str, bib: Path, tex: Path, out: Path = OUT) -> None:
    cmd = [
        exe, str(tex), "-f", "latex+raw_tex", "-t", "docx+native_numbering",
        "--citeproc", "--bibliography", str(bib), "--csl", str(CSL),
        "-M", "reference-section-title=References",
        "--lua-filter", str(LUA_FILTER),  # after --citeproc: also sees the reference list
        "--reference-doc", str(REFERENCE),
        "--number-sections", "--table-caption-position=below",
        "-o", str(out),
    ]
    res = subprocess.run(cmd, cwd=REPORT_DIR, capture_output=True, text=True, encoding="utf-8")
    if res.stderr.strip():
        print(res.stderr.strip(), file=sys.stderr)
    if res.returncode != 0:
        hint = " (is report.docx open in Word? close it and retry)" if "denied" in res.stderr else ""
        sys.exit(f"pandoc failed with exit code {res.returncode}{hint}")


# --------------------------------------------------------------------------- template
def _fragment(xml: str) -> list:
    return list(parse_xml(f"<w:root {nsdecls('w')}>{xml}</w:root>"))


def fonts(name: str = FONT) -> str:
    return f'<w:rFonts w:ascii="{name}" w:hAnsi="{name}" w:eastAsia="{name}" w:cs="{name}"/>'


def size(pt: float) -> str:
    v = round(pt * 2)
    return f'<w:sz w:val="{v}"/><w:szCs w:val="{v}"/>'


def spacing(before: int = 0, after: int = 0, line: int = 240) -> str:  # twips
    return f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'


BOLD = "<w:b/><w:bCs/>"
KEEP = "<w:keepNext/><w:keepLines/>"


def set_style(styles, sid, name, kind="paragraph", based_on=None, next_=None,
              ppr="", rpr="", tblpr="", extra="", custom=False):
    """Create style `sid`, or update it in place. Formatting (pPr/rPr/tblPr/...) is replaced
    as a whole; basedOn/next only when given. Child order follows the OOXML schema."""
    st = next((s for s in styles.iterfind(qn("w:style")) if s.get(qn("w:styleId")) == sid), None)
    if st is None:
        st = parse_xml(f'<w:style {nsdecls("w")} w:type="{kind}" w:styleId="{sid}">'
                       f'<w:name w:val="{name}"/><w:qFormat/></w:style>')
        if custom:
            st.set(qn("w:customStyle"), "1")
        styles.append(st)
    drop = ["pPr", "rPr", "tblPr", "trPr", "tcPr", "tblStylePr"]
    drop += [t for t, v in (("basedOn", based_on), ("next", next_)) if v]
    for tag in drop:
        for el in st.findall(qn(f"w:{tag}")):
            st.remove(el)
    anchor = st.find(qn("w:aliases"))
    anchor = anchor if anchor is not None else st.find(qn("w:name"))
    for tag, val in (("next", next_), ("basedOn", based_on)):  # inserted in reverse
        if val:
            anchor.addnext(parse_xml(f'<w:{tag} {nsdecls("w")} w:val="{val}"/>'))
    body = (f"<w:pPr>{ppr}</w:pPr>" if ppr else "") + (f"<w:rPr>{rpr}</w:rPr>" if rpr else "")
    body += (f"<w:tblPr>{tblpr}</w:tblPr>" if tblpr else "") + extra
    for el in _fragment(body):
        st.append(el)


def style_document(styles) -> None:
    dd = styles.find(qn("w:docDefaults"))
    for tag, xml in (("w:rPrDefault", f'<w:rPr>{fonts()}{size(12)}'
                      '<w:lang w:val="en-GB" w:eastAsia="zh-CN" w:bidi="ar-SA"/></w:rPr>'),
                     ("w:pPrDefault", f"<w:pPr><w:widowControl/>{spacing()}</w:pPr>")):
        el = dd.find(qn(tag))
        for child in list(el):
            el.remove(child)
        el.extend(_fragment(xml))

    set_style(styles, "BodyText", "Body Text", based_on="Normal",
              ppr=spacing(0, 0, LINE_1_5) + '<w:ind w:firstLine="360"/><w:jc w:val="both"/>')
    set_style(styles, "FirstParagraph", "First Paragraph", based_on="BodyText", next_="BodyText",
              ppr='<w:ind w:firstLine="0"/>')
    set_style(styles, "RunInParagraph", "Run-in Paragraph", based_on="BodyText", next_="BodyText",
              ppr=spacing(240, 0, LINE_1_5) + '<w:ind w:firstLine="0"/>', custom=True)
    set_style(styles, "ListParagraph", "List Paragraph", based_on="Normal",
              ppr=spacing(0, 80, LINE_1_5) + '<w:jc w:val="both"/>')
    # table cells (LaTeX \small) and the \footnotesize variant
    set_style(styles, "Compact", "Compact", based_on="Normal", ppr=spacing(), rpr=size(11))
    set_style(styles, "TableTextSmall", "Table Text Small", kind="character",
              based_on="DefaultParagraphFont", rpr=size(10), custom=True)
    set_style(styles, "TableCaption", "Table Caption", based_on="Caption",
              ppr=spacing(120, 240) + '<w:jc w:val="center"/>',
              rpr='<w:i w:val="0"/><w:iCs w:val="0"/>' + size(11))

    # title page: \LARGE title lines, \large author and date (\vskip 1.5em / 1em before them)
    set_style(styles, "Title", "Title", based_on="Normal", next_="BodyText",
              ppr=spacing() + '<w:jc w:val="center"/>', rpr=fonts() + size(20.5))
    set_style(styles, "Author", "Author", based_on="Title", next_="BodyText",
              ppr=KEEP + spacing(360), rpr=size(14.5))
    set_style(styles, "Date", "Date", based_on="Title", next_="BodyText",
              ppr=KEEP + spacing(240), rpr=size(14.5))

    # headings: \section \Large, \subsection \large, \subsubsection normal size; all bold, black
    heads = {1: (17.5, 360, 200), 2: (14.5, 300, 120), 3: (12, 280, 120)}
    for lvl in range(1, 10):
        pt, before, after = heads.get(lvl, (12, 240, 0))
        set_style(styles, f"Heading{lvl}", f"heading {lvl}", based_on="Normal", next_="BodyText",
                  ppr=KEEP + spacing(before, after) + f'<w:outlineLvl w:val="{lvl - 1}"/>',
                  rpr=fonts() + BOLD + size(pt))
    # \section* and friends: same look, kept out of the table of contents (as in LaTeX)
    for lvl in (1, 2, 3):
        set_style(styles, f"UnnumberedHeading{lvl}", f"Unnumbered Heading {lvl}",
                  based_on=f"Heading{lvl}", next_="BodyText",
                  ppr='<w:outlineLvl w:val="9"/>', custom=True)
    set_style(styles, "TOCHeading", "TOC Heading", based_on="Heading1", next_="BodyText",
              ppr='<w:outlineLvl w:val="9"/>')
    set_style(styles, "TOC1", "toc 1", based_on="Normal", next_="Normal",
              ppr=spacing(120, 0, 300), rpr=BOLD)
    for lvl, left in ((2, 360), (3, 840)):
        set_style(styles, f"TOC{lvl}", f"toc {lvl}", based_on="Normal", next_="Normal",
                  ppr=spacing(0, 0, 300) + f'<w:ind w:left="{left}"/>')

    # "[n]<tab>entry" with a hanging indent, like thebibliography
    set_style(styles, "Bibliography", "Bibliography", based_on="Normal", next_="Bibliography",
              ppr='<w:tabs><w:tab w:val="left" w:pos="567"/></w:tabs>' + spacing(0, 60, 300)
              + '<w:ind w:left="567" w:hanging="567"/>')
    set_style(styles, "Hyperlink", "Hyperlink", kind="character",
              rpr='<w:color w:val="0563C1"/><w:u w:val="single"/>')
    set_style(styles, "Footer", "footer", based_on="Normal", ppr='<w:jc w:val="center"/>')

    # booktabs look: heavy top and bottom rules, light rule under the header row; centred
    rule = '<w:{side} w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>'
    set_style(styles, "Table", "Table", kind="table",
              tblpr='<w:jc w:val="center"/><w:tblInd w:w="0" w:type="dxa"/><w:tblBorders>'
                    + rule.format(side="top", sz=8) + rule.format(side="bottom", sz=8)
                    + '</w:tblBorders><w:tblCellMar><w:top w:w="40" w:type="dxa"/>'
                    '<w:left w:w="108" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/>'
                    '<w:right w:w="108" w:type="dxa"/></w:tblCellMar>',
              extra='<w:tblStylePr w:type="firstRow"><w:tcPr><w:tcBorders>'
                    + rule.format(side="bottom", sz=4)
                    + '</w:tcBorders><w:vAlign w:val="bottom"/></w:tcPr></w:tblStylePr>')


def build_reference(exe: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        default = Path(tmp) / "default.docx"
        subprocess.run([exe, "-o", str(default), "--print-default-data-file", "reference.docx"],
                       check=True)
        doc = Document(io.BytesIO(default.read_bytes()))

    style_document(doc.styles.element)

    sec = doc.sections[0]  # LaTeX: a4paper; page number centred in the footer
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.54)
    sec.header_distance = sec.footer_distance = Cm(1.25)
    par = sec.footer.paragraphs[0]
    par.style = doc.styles["footer"]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par._p.append(parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE">'
                            "<w:r><w:t>1</w:t></w:r></w:fldSimple>"))

    settings = doc.settings.element
    if settings.find(qn("w:compat")) is None:  # open as a current Word file, not compatibility mode
        compat = parse_xml(f'<w:compat {nsdecls("w")}><w:compatSetting w:name="compatibilityMode" '
                           'w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>')
        anchor = settings.find(qn("w:rsids"))
        anchor.addprevious(compat) if anchor is not None else settings.append(compat)
    doc.save(str(REFERENCE))


# --------------------------------------------------------------------------- finishing
def finish(path: Path) -> None:
    doc = Document(str(path))
    for tbl in doc.tables:  # a table and its caption (below it) stay on one page
        for row in tbl.rows:
            tr_pr = row._tr.get_or_add_trPr()
            if tr_pr.find(qn("w:cantSplit")) is None:
                tr_pr.append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True

    def texts(style_id):
        return [p.text.strip() for p in doc.paragraphs
                if p.style is not None and p.style.style_id == style_id and p.text.strip()]

    props, titles, authors = doc.core_properties, texts("Title"), texts("Author")
    if titles:
        props.title = ": ".join(titles[:2])
    if authors:
        props.author = "; ".join(authors)
    settings = doc.settings.element  # pandoc marks the text as already spell-checked
    for el in settings.findall(qn("w:proofState")):
        settings.remove(el)
    doc.save(str(path))


def word_update(path: Path, pdf: Path | None) -> bool:
    """Fill in the table of contents (and SEQ table numbers) with Word, then save."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        print("pywin32 not installed: skipped the Word step; the contents fill in on first open.")
        return False
    pythoncom.CoInitialize()
    try:
        try:
            word = win32com.client.DispatchEx("Word.Application")  # own instance, never the user's
        except Exception as exc:  # noqa: BLE001 - COM raises pywintypes.com_error
            print(f"Microsoft Word not available ({exc}); the contents fill in on first open.")
            return False
        try:
            word.Visible = False
            word.DisplayAlerts = 0
            doc = word.Documents.Open(str(path), False, False, False)
            try:
                doc.Fields.Update()
                for i in range(1, doc.TablesOfContents.Count + 1):
                    doc.TablesOfContents(i).Update()
                doc.Save()
                if pdf:
                    doc.ExportAsFixedFormat(str(pdf.resolve()), 17)  # wdExportFormatPDF
            finally:
                doc.Close(0)
        finally:
            word.Quit()
    finally:
        pythoncom.CoUninitialize()
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert report.tex to an editable report.docx.")
    ap.add_argument("--no-word", action="store_true", help="skip the Microsoft Word step")
    ap.add_argument("--out", type=Path, default=OUT, help="DOCX output path")
    ap.add_argument("--pdf", type=Path, help="also export a PDF through Word (layout check)")
    args = ap.parse_args()

    output = args.out if args.out.is_absolute() else (REPORT_DIR / args.out)
    output = output.resolve()
    exe = find_pandoc()
    build_reference(exe)
    bib = bibliography_file()
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)
        safe_bib = temporary / bib.name
        safe_bib.write_text(protect_case(bib.read_text(encoding="utf-8")), encoding="utf-8")
        expanded_tex = temporary / "report-expanded.tex"
        expanded_tex.write_text(prepare_pandoc_tex(TEX), encoding="utf-8")
        run_pandoc(exe, safe_bib, expanded_tex, output)
    finish(output)
    used_word = False if args.no_word else word_update(output, args.pdf)
    print(f"wrote {output}" + ("" if used_word else " (contents: right-click > Update Field in Word)"))


if __name__ == "__main__":
    main()
