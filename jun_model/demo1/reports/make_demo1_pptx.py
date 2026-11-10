from __future__ import annotations

import html
import os
import zipfile
from pathlib import Path


OUT = Path("jun_model/demo1/reports/demo1_presentation.pptx")

SLIDES = [
    {
        "title": "Scanner Generalization for Mitotic Figure Detection",
        "subtitle": "Demo 1 - CSE 4109 AI for Health",
        "bullets": [
            "Robust mitotic figure detection on real breast cancer pathology images",
            "Focus: scanner-domain shift in MIDOG++",
            "Demo goal: show real dataset, working pipeline, and early results",
        ],
        "image": None,
    },
    {
        "title": "Why This Matters",
        "subtitle": None,
        "bullets": [
            "Pathology AI models can fail when image appearance changes across scanners.",
            "Mitotic figures are clinically important for cancer grading.",
            "Scanner and stain differences affect color, contrast, and detector confidence.",
            "Research question: can a detector trained on some scanners generalize to an unseen scanner?",
        ],
        "image": "jun_model/demo1/outputs/scanner_stain_comparison.jpg",
    },
    {
        "title": "Dataset And Task",
        "subtitle": "MIDOG++ human breast cancer subset",
        "bullets": [
            "150 TIFF images and 4435 object annotations",
            "Three scanner domains: Aperio CS2, Hamamatsu S360, Hamammatsu XR",
            "Labels are COCO-style bounding boxes for mitotic and non-mitotic candidates",
            "Prepared shareable subset with images, metadata, labels, and README",
        ],
        "image": "jun_model/demo1/outputs/001_annotations.jpg",
    },
    {
        "title": "Experimental Design",
        "subtitle": "Leave-one-scanner-out evaluation",
        "bullets": [
            "Train on two scanner domains.",
            "Test on the held-out scanner domain.",
            "Keeps tumor type fixed while testing scanner generalization.",
            "Pipeline: annotations -> breast cancer subset -> detection tiles -> Faster R-CNN -> held-out scanner metrics",
        ],
        "image": None,
    },
    {
        "title": "What We Built For Demo 1",
        "subtitle": None,
        "bullets": [
            "Breast cancer subset and teammate-facing metadata documentation",
            "Patch-level scanner robustness experiments",
            "512 x 512 detection tile datasets for all scanners",
            "Faster R-CNN training, evaluation, checkpoint sweep scripts",
            "Visual false positive / false negative comparison panels",
            "First stain normalization experiment",
        ],
        "image": "jun_model/demo1/outputs/detection_tile_positive_preview.jpg",
    },
    {
        "title": "Initial Patch-Level LOSO Results",
        "subtitle": "Logistic regression on cropped candidates: baseline vs color augmentation",
        "table": [
            ["Held-out scanner", "Base P", "Base R", "Base F1", "Aug P", "Aug R", "Aug F1", "Delta F1"],
            ["Aperio CS2", "0.444", "0.384", "0.412", "0.460", "0.440", "0.450", "+0.038"],
            ["Hamamatsu S360", "0.346", "0.345", "0.346", "0.372", "0.466", "0.413", "+0.068"],
            ["Hamammatsu XR", "0.396", "0.794", "0.529", "0.395", "0.785", "0.525", "-0.004"],
            ["Mean", "0.395", "0.508", "0.429", "0.409", "0.564", "0.463", "+0.034"],
        ],
        "takeaway": "Augmentation improved mean F1 from 0.429 to 0.463 and mean recall from 0.508 to 0.564.",
        "image": None,
    },
    {
        "title": "Full Detection Results",
        "subtitle": "Faster R-CNN on pathology detection tiles",
        "bullets": [
            "Stable baseline mean F1: 0.728",
            "Strong color jitter mean F1: 0.698",
            "Stable training and checkpoint selection mattered a lot.",
            "Takeaway: color jitter helped patches but did not improve full object detection.",
        ],
        "image": None,
    },
    {
        "title": "Normalization Result",
        "subtitle": "Held-out Hamamatsu S360",
        "bullets": [
            "Stable baseline: precision 0.633, recall 0.817, F1 0.714",
            "Reinhard RGB normalization: precision 0.709, recall 0.721, F1 0.715",
            "Normalization reduced false positives: 379 -> 237",
            "It preserved F1 but became more conservative.",
        ],
        "image": None,
    },
    {
        "title": "Visual Error Analysis",
        "subtitle": "Example comparison panels",
        "bullets": [
            "Green boxes: ground truth",
            "Red boxes: baseline predictions",
            "Blue boxes: comparison model predictions",
            "Panels reveal false positives, false negatives, and scanner-specific failure modes.",
        ],
        "image": "jun_model/demo1/outputs/detection_error_panels_s360_baseline_vs_mild/comparison_fn_extra_01.jpg",
    },
    {
        "title": "Interpretation And Next Steps",
        "subtitle": None,
        "bullets": [
            "LOSO gives a real scanner-generalization test.",
            "Patch-level improvement did not automatically transfer to full detection.",
            "Simple normalization is more promising than stronger color jitter so far.",
            "Next: run normalization on all scanners, make baseline-vs-normalization panels, and try Macenko-style stain normalization.",
        ],
        "image": None,
    },
]


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def text_box(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, text: str, size: int, bold: bool = False) -> str:
    bold_xml = "<a:b/>" if bold else ""
    return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
        <p:txBody>
          <a:bodyPr wrap="square"/>
          <a:lstStyle/>
          <a:p><a:r><a:rPr lang="en-US" sz="{size}">{bold_xml}<a:solidFill><a:srgbClr val="1F2937"/></a:solidFill></a:rPr><a:t>{esc(text)}</a:t></a:r><a:endParaRPr lang="en-US" sz="{size}"/></a:p>
        </p:txBody>
      </p:sp>
    """


def bullet_box(shape_id: int, bullets: list[str], has_image: bool) -> str:
    x = 650000
    y = 1750000
    cx = 5200000 if has_image else 10800000
    cy = 4200000
    paragraphs = []
    for bullet in bullets:
        paragraphs.append(
            f"""
            <a:p>
              <a:pPr marL="285750" indent="-171450"><a:buChar char="&#8226;"/></a:pPr>
              <a:r><a:rPr lang="en-US" sz="2450"><a:solidFill><a:srgbClr val="111827"/></a:solidFill></a:rPr><a:t>{esc(bullet)}</a:t></a:r>
            </a:p>
            """
        )
    return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="Bullets"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
        <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{''.join(paragraphs)}</p:txBody>
      </p:sp>
    """


def image_shape(shape_id: int, rel_id: str) -> str:
    return f"""
      <p:pic>
        <p:nvPicPr><p:cNvPr id="{shape_id}" name="Result image"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr><a:xfrm><a:off x="6400000" y="1700000"/><a:ext cx="5600000" cy="3900000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
      </p:pic>
    """


def table_cell(text: str, header: bool = False, highlight: bool = False) -> str:
    fill = "2563EB" if header else ("E0F2FE" if highlight else "FFFFFF")
    color = "FFFFFF" if header else "111827"
    bold = "<a:b/>" if header or highlight else ""
    return f"""
      <a:tc>
        <a:txBody>
          <a:bodyPr anchor="ctr"/><a:lstStyle/>
          <a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="1450">{bold}<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr><a:t>{esc(text)}</a:t></a:r><a:endParaRPr lang="en-US" sz="1450"/></a:p>
        </a:txBody>
        <a:tcPr marL="70000" marR="70000" marT="50000" marB="50000"><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:lnL w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnL><a:lnR w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnR><a:lnT w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnT><a:lnB w="6350"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:lnB></a:tcPr>
      </a:tc>
    """


def table_shape(shape_id: int, rows: list[list[str]]) -> str:
    widths = [2300000, 1250000, 1250000, 1250000, 1250000, 1250000, 1250000, 1200000]
    grid = "".join(f'<a:gridCol w="{width}"/>' for width in widths)
    row_xml = []
    for row_index, row in enumerate(rows):
        cells = "".join(
            table_cell(value, header=row_index == 0, highlight=row_index == len(rows) - 1)
            for value in row
        )
        row_xml.append(f'<a:tr h="650000">{cells}</a:tr>')
    return f"""
      <p:graphicFrame>
        <p:nvGraphicFramePr><p:cNvPr id="{shape_id}" name="Patch LOSO results table"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>
        <p:xfrm><a:off x="595000" y="1700000"/><a:ext cx="11000000" cy="3250000"/></p:xfrm>
        <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">
          <a:tbl><a:tblPr firstRow="1" bandRow="1"/><a:tblGrid>{grid}</a:tblGrid>{''.join(row_xml)}</a:tbl>
        </a:graphicData></a:graphic>
      </p:graphicFrame>
    """


def slide_xml(slide: dict, n: int) -> str:
    has_image = bool(slide.get("image") and Path(slide["image"]).exists())
    shapes = [
        text_box(2, "Title", 650000, 350000, 11200000, 650000, slide["title"], 3600, True),
    ]
    if slide.get("subtitle"):
        shapes.append(text_box(3, "Subtitle", 700000, 1020000, 11000000, 420000, slide["subtitle"], 1900, False))
    if slide.get("table"):
        shapes.append(table_shape(4, slide["table"]))
        shapes.append(text_box(5, "Takeaway", 750000, 5350000, 10700000, 700000, slide["takeaway"], 2050, True))
    else:
        shapes.append(bullet_box(4, slide["bullets"], has_image))
    if has_image:
        shapes.append(image_shape(6 if slide.get("table") else 5, "rId2"))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def slide_rels(slide: dict) -> str:
    image = slide.get("image")
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    if image and Path(image).exists():
        ext = Path(image).suffix.lower().lstrip(".")
        rels.append(
            f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{Path(image).name}"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>
"""


def presentation_xml() -> str:
    ids = []
    for idx in range(1, len(SLIDES) + 1):
        ids.append(f'<p:sldId id="{255 + idx}" r:id="rId{idx + 2}"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{''.join(ids)}</p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def presentation_rels() -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    ]
    for idx in range(1, len(SLIDES) + 1):
        rels.append(
            f'<Relationship Id="rId{idx + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{idx}.xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>
"""


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
""" + "".join(
    f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>\n'
    for i in range(1, len(SLIDES) + 1)
) + "</Types>\n"


ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


CORE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Demo 1 Presentation</dc:title>
  <dc:subject>MIDOG++ scanner generalization</dc:subject>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>
"""


APP = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""


THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="DemoTheme">
  <a:themeElements>
    <a:clrScheme name="Demo"><a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="F8FAFC"/></a:lt1><a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="FFFFFF"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="059669"/></a:accent2><a:accent3><a:srgbClr val="DC2626"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="0891B2"/></a:accent5><a:accent6><a:srgbClr val="EA580C"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Demo"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Demo"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>
"""


MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>
"""


MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def write_pptx() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("docProps/core.xml", CORE)
        z.writestr("docProps/app.xml", APP)
        z.writestr("ppt/presentation.xml", presentation_xml())
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels())
        z.writestr("ppt/theme/theme1.xml", THEME)
        z.writestr("ppt/slideMasters/slideMaster1.xml", MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", LAYOUT_RELS)

        copied_media: set[str] = set()
        for idx, slide in enumerate(SLIDES, 1):
            z.writestr(f"ppt/slides/slide{idx}.xml", slide_xml(slide, idx))
            z.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels(slide))
            image = slide.get("image")
            if image and Path(image).exists():
                image_path = Path(image)
                arc = f"ppt/media/{image_path.name}"
                if arc not in copied_media:
                    z.write(image_path, arc)
                    copied_media.add(arc)


if __name__ == "__main__":
    write_pptx()
    print(f"wrote {OUT}")
