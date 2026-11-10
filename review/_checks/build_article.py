"""Сборка оформленной статьи review/ARTICLE.docx (python-docx + формулы OMML).

Запуск из корня проекта:  python -I -B -X utf8 review\\_checks\\build_article.py review
Текст статьи задан в этом файле. Формулы пишутся в $...$ на упрощённом LaTeX.
"""
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

REVIEW = Path(sys.argv[1])
FONT = "Times New Roman"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# ---------------------------------------------------------------- LaTeX -> OMML
SYMBOLS = {
    "nu": "ν", "tau": "τ", "pi": "π", "varphi": "φ", "theta": "θ", "Omega": "Ω", "infty": "∞",
    "cdot": "·", "to": "→", "approx": "≈", "le": "≤", "ge": "≥", "ll": "≪", "gg": "≫",
    "pm": "±", "mp": "∓", "times": "×", "prime": "′", "quad": " ", ",": " ", "sim": "∼",
}


class Tex:
    def __init__(self, source: str, size_pt: float):
        self.s, self.i, self.size = source, 0, int(size_pt * 2)

    def run(self, text: str, style: str | None = None) -> str:
        sty = f'<m:rPr><m:sty m:val="{style}"/></m:rPr>' if style else ""
        return (f"<m:r>{sty}<w:rPr><w:rFonts w:ascii=\"Cambria Math\" w:hAnsi=\"Cambria Math\"/>"
                f"<w:sz w:val=\"{self.size}\"/></w:rPr>"
                f"<m:t xml:space=\"preserve\">{escape(text)}</m:t></m:r>")

    def raw_group(self) -> str:
        assert self.s[self.i] == "{", self.s
        end = self.s.index("}", self.i)
        text = self.s[self.i + 1:end]
        self.i = end + 1
        return text

    def arg(self) -> str:
        if self.i < len(self.s) and self.s[self.i] == "{":
            self.i += 1
            inner = self.seq("}")
            self.i += 1
            return inner
        return self.atom()

    def scripts(self) -> tuple[str | None, str | None]:
        sub = sup = None
        while self.i < len(self.s) and self.s[self.i] in "_^":
            kind = self.s[self.i]
            self.i += 1
            if kind == "_":
                sub = self.arg()
            else:
                sup = self.arg()
        return sub, sup

    def seq(self, stop: str | None = None) -> str:
        out = []
        while self.i < len(self.s) and not (stop and self.s.startswith(stop, self.i)):
            if self.s.startswith("\\int", self.i):
                self.i += 4
                sub, sup = self.scripts()
                body = self.seq(stop)
                hide = "".join(f'<m:{n}Hide m:val="1"/>' for n, v in (("sub", sub), ("sup", sup)) if v is None)
                out.append(
                    f'<m:nary><m:naryPr><m:chr m:val="∫"/><m:limLoc m:val="subSup"/>{hide}</m:naryPr>'
                    f"<m:sub>{sub or ''}</m:sub><m:sup>{sup or ''}</m:sup><m:e>{body}</m:e></m:nary>")
                continue
            base = self.atom()
            sub, sup = self.scripts()
            if sub and sup:
                base = f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
            elif sub:
                base = f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
            elif sup:
                base = f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"
            out.append(base)
        return "".join(out)

    def atom(self) -> str:
        c = self.s[self.i]
        if c == "{":
            self.i += 1
            inner = self.seq("}")
            self.i += 1
            return inner
        if c == "\\":
            match = re.match(r"\\([A-Za-z]+|.)", self.s[self.i:])
            name = match.group(1)
            self.i += len(match.group(0))
            if name == "frac":
                num, den = self.arg(), self.arg()
                return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"
            if name in ("mathrm", "text"):
                return self.run(self.raw_group(), "p")
            if name == "mathbf":
                return self.run(self.raw_group(), "b")
            if name == "left":
                self.i += 1  # открывающая скобка
                inner = self.seq("\\right")
                self.i += len("\\right") + 1
                return f"<m:d><m:e>{inner}</m:e></m:d>"
            return self.run(SYMBOLS[name])
        if c == " ":
            self.i += 1
            return ""
        number = re.match(r"\d+(?:[,.]\d+)?", self.s[self.i:])
        if number:
            self.i += len(number.group(0))
            return self.run(number.group(0))
        self.i += 1
        if c == "-":
            c = "−"
        return self.run(c, "p" if "а" <= c.lower() <= "я" else None)


def omath(source: str, size_pt: float):
    body = Tex(source, size_pt).seq()
    return parse_xml(f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W_NS}">{body}</m:oMath>')


# ---------------------------------------------------------------- документ
doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(section, side, Cm(2))
TEXT_WIDTH = Cm(17)

normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(14)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
normal.element.rPr.rFonts.set(qn("w:cs"), FONT)

INLINE = re.compile(r"(\$[^$]+\$|\*\*[^*]+\*\*|\[\[.+?\]\])")
counters = {"eq": 0, "fig": 0, "tab": 0}


def fmt(par, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, spacing=1.5, before=0, after=0, keep=False):
    f = par.paragraph_format
    par.alignment = align
    f.first_line_indent = Cm(1.25) if indent else Cm(0)
    f.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    f.line_spacing = spacing
    f.space_before, f.space_after = Pt(before), Pt(after)
    f.keep_with_next = keep
    return par


def rich(par, text: str, size: float = 14, bold: bool = False, italic: bool = False):
    """Текст с $формулами$, **жирным** и [[жёлтой пометкой]]."""
    parts = [part for part in INLINE.split(text) if part]
    # Знак препинания сразу после формулы переносим внутрь неё: иначе Word
    # при выравнивании по ширине отрывает его пробелом.
    for index in range(len(parts) - 1):
        following = parts[index + 1]
        if parts[index].startswith("$") and following[0] in ".,;:" and not following.startswith("$"):
            parts[index] = parts[index][:-1] + "\\mathrm{" + following[0] + "}$"
            parts[index + 1] = following[1:]
    for part in parts:
        if not part:
            continue
        if part.startswith("$"):
            par._p.append(omath(part[1:-1], size))
            continue
        marked = part.startswith("[[")
        strong = part.startswith("**")
        run = par.add_run(part[2:-2] if marked or strong else part)
        run.font.size = Pt(size)
        run.bold = bold or strong
        run.italic = italic
        if marked:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return par


def text(content: str, **kw):
    return rich(fmt(doc.add_paragraph(), **kw), content)


def heading(content: str):
    par = fmt(doc.add_paragraph(), align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, before=12, after=6, keep=True)
    return rich(par, content, bold=True)


def equation(source: str) -> int:
    """Выключная формула: таблица без рамок [пусто | формула | (номер)]."""
    counters["eq"] += 1
    t = doc.add_table(rows=1, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for cell, width in zip(t.rows[0].cells, (1.5, 14.0, 1.5)):
        cell.width = Cm(width)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    middle = fmt(t.cell(0, 1).paragraphs[0], align=WD_ALIGN_PARAGRAPH.CENTER, indent=False,
                 spacing=1.0, before=5, after=5)
    para = parse_xml(f'<m:oMathPara xmlns:m="{M_NS}"/>')
    para.append(omath(source, 14))
    middle._p.append(para)
    right = fmt(t.cell(0, 2).paragraphs[0], align=WD_ALIGN_PARAGRAPH.RIGHT, indent=False, spacing=1.0)
    right.add_run(f"({counters['eq']})")
    return counters["eq"]


def figure(image: str, caption: str):
    counters["fig"] += 1
    par = fmt(doc.add_paragraph(), align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, spacing=1.0, before=6, keep=True)
    par.add_run().add_picture(str(REVIEW / "figures" / image), width=Cm(15.5))
    cap = fmt(doc.add_paragraph(), align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, spacing=1.0, after=10)
    rich(cap, f"**Рис. {counters['fig']}.** {caption}", size=12)


def table(caption: str, rows: list[list[str]], widths_cm: list[float]):
    counters["tab"] += 1
    cap = fmt(doc.add_paragraph(), align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, spacing=1.0, before=8, after=4, keep=True)
    rich(cap, f"**Таблица {counters['tab']}.** {caption}", size=12)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for ri, row in enumerate(rows):
        for ci, value in enumerate(row):
            cell = t.cell(ri, ci)
            cell.width = Cm(widths_cm[ci])
            par = fmt(cell.paragraphs[0], indent=False, spacing=1.0, before=2, after=2,
                      align=WD_ALIGN_PARAGRAPH.CENTER if ci or ri == 0 else WD_ALIGN_PARAGRAPH.LEFT)
            rich(par, value, size=12, bold=ri == 0)
    fmt(doc.add_paragraph(), indent=False, spacing=1.0, after=4)


def centered(content: str, size=14, bold=False, italic=False, after=0):
    par = fmt(doc.add_paragraph(), align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, spacing=1.15, after=after)
    return rich(par, content, size=size, bold=bold, italic=italic)


def small(content: str, after=0):
    return rich(fmt(doc.add_paragraph(), spacing=1.15, after=after), content, size=12)



def F(template: str, *numbers) -> str:
    """Подставляет номера формул вместо ({}) — без str.format, чтобы не мешали скобки LaTeX."""
    for number in numbers:
        template = template.replace("({})", f"({number})", 1)
    return template


# ================================================================ ТЕКСТ СТАТЬИ
rich(fmt(doc.add_paragraph(), align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, after=10), "УДК 536.3:533.9")

centered("ТОЧНОСТЬ ПРИБЛИЖЕНИЯ ШУСТЕРА — ШВАРЦШИЛЬДА ПРИ РАСЧЁТЕ ПЕРЕНОСА ИЗЛУЧЕНИЯ "
         "В ЦИЛИНДРИЧЕСКОМ СТОЛБЕ КСЕНОНОВОЙ ПЛАЗМЫ С НЕОДНОРОДНЫМ ПО СПЕКТРУ ПОГЛОЩЕНИЕМ",
         bold=True, after=10)
centered("[[И. О. Фамилия¹, И. О. Фамилия¹ (научный руководитель)]]", after=4)
centered("[[¹ МГТУ им. Н. Э. Баумана, Москва, Россия; e-mail: адрес@student.bmstu.ru]]", size=12, italic=True, after=12)

small("**Аннотация.** Исследована точность двухпоточного приближения Шустера — Шварцшильда при расчёте "
      "спектральной дивергенции лучистого потока в бесконечном цилиндре с заданным профилем температуры. "
      "Среда — ксенон при давлении 1,5 МПа. Коэффициент поглощения задан на 193 спектральных полосах, "
      "радиальная оптическая толщина которых меняется от 5·10⁻⁴ до 2,4·10³. Эталоном служит прямое "
      "интегрирование уравнения переноса вдоль лучей. Показано, что погрешность приближения определяется "
      "оптической толщиной полосы. В прозрачных полосах (τ < 0,1) отличие от точного решения не превышает "
      "нескольких процентов. В оптически толстых полосах отношение приближённой дивергенции к точной "
      "стремится к 3/4 вне оси и к 3/8 на оси; оба предела получены аналитически и подтверждены расчётом. "
      "Для интегральной по спектру дивергенции погрешность составила 11 % на оси и 2–6 % в остальном объёме, "
      "для потока с поверхности — 3,3 %.", after=6)
small("**Ключевые слова:** перенос излучения, приближение Шустера — Шварцшильда, двухпоточное приближение, "
      "дивергенция лучистого потока, оптическая толщина, диффузионный предел, ксеноновая плазма.", after=12)

# ---------------------------------------------------------------- 1
heading("Введение")
text("При моделировании сильноточных разрядов в инертных газах уравнение энергии содержит дивергенцию "
     "лучистого потока. Она определяет радиационные потери горячего ядра разряда и нагрев периферийных слоёв "
     "собственным излучением плазмы [1, 2]. Точное вычисление этой величины требует интегрирования уравнения "
     "переноса по направлениям и по спектру в каждой точке, что дорого в итерационных расчётах. Поэтому на "
     "практике применяют приближённые модели: диффузионную и двухпоточные [2–4].")
text("Приближение Шустера — Шварцшильда [5, 6] заменяет угловое распределение интенсивности двумя "
     "полусферическими потоками. Для цилиндрически симметричных течений излучающего газа алгоритм на его "
     "основе описан в работе [7], где он сопоставлен с методами диффузионного типа. Погрешность двухпоточной "
     "модели зависит от оптической толщины среды. В плазме с линейчатым спектром, в частности в ксеноновой [8], "
     "оптическая толщина на разных частотах различается на много порядков. Поэтому одна и та же модель "
     "одновременно работает в прозрачном, промежуточном и диффузионном режимах.")
text("Цель работы — оценить погрешность приближения Шустера — Шварцшильда для цилиндра с заданным полем "
     "температур и реальным спектром поглощения ксенона, сравнив его с прямым решением уравнения переноса.")

# ---------------------------------------------------------------- 2
heading("1. Постановка задачи")
text("Рассматривается бесконечный осесимметричный цилиндр радиуса $R$. Температура задана выражением")
eq_T = equation(r"T(r)=T_0+(T_w-T_0)\left(\frac{r}{R}\right)^n,")
text("где $T_0$ и $T_w$ — температуры на оси и у боковой поверхности. Среда излучает и поглощает, рассеяние "
     "не учитывается, показатель преломления равен единице. Принято локальное термодинамическое равновесие: "
     "коэффициент излучения $j_\\nu=k_\\nu B_\\nu(T)$, где $k_\\nu$ — коэффициент поглощения, $B_\\nu$ — функция "
     "Планка [1]. Внешнее излучение на границе отсутствует.", indent=False)
text("Коэффициент поглощения ксенона $k_\\nu(T)$ при давлении 1,5 МПа задан таблицей: 193 полосы в диапазоне "
     "частот от $0,02\\cdot 10^{15}$ до $3,0\\cdot 10^{15}$ Гц и 12 температур от 2000 до 13 000 К. Внутри полосы $k_\\nu$ не "
     "зависит от частоты. По температуре использована линейная интерполяция; ниже 2000 К принято "
     "$k_\\nu(T)=k_\\nu(2000\\,\\mathrm{К})$. [[Табличные значения приняты заданными в м⁻¹ — уточнить единицы.]]")
text("Искомая величина — спектральная дивергенция лучистого потока $q_\\nu(r)=\\mathrm{div}\\,F_\\nu$ и её "
     "интеграл по спектру $Q(r)$, входящий в уравнение энергии. Положительные значения соответствуют потерям "
     "энергии веществом. Характеристикой полосы служит радиальная оптическая толщина")
eq_tau = equation(r"\tau_\nu=\int_0^R k_\nu\left(T(r)\right)\,dr.")

# ---------------------------------------------------------------- 3
heading("2. Методы расчёта")
heading("2.1. Прямое интегрирование уравнения переноса")
text("Интенсивность $I_\\nu$ в точке $r$ в направлении $\\mathbf{Ω}$ находится из уравнения переноса вдоль луча "
     "от поверхности до точки наблюдения [1, 3]:")
eq_rte = equation(r"\frac{dI_\nu}{ds}=k_\nu\left(B_\nu-I_\nu\right),\quad I_\nu=0\mathrm{ на входе луча}.")
text(F("Луч разбивается на отрезки, сгущающиеся к точке наблюдения. На каждом отрезке $k_\\nu$ и $B_\\nu$ "
     "постоянны, и уравнение ({}) интегрируется аналитически. Результирующий поток и дивергенция получаются "
     "интегрированием по телесному углу:", eq_rte))
eq_flux = equation(r"F_\nu=F_\nu^+-F_\nu^-=\int_{4\pi} I_\nu\,(\mathbf{Ω}\cdot\mathbf{e}_r)\,d\mathrm{Ω},")
eq_qex = equation(r"q_\nu=k_\nu\int_{4\pi}\left(B_\nu-I_\nu\right)d\mathrm{Ω}.")
text("Интегралы по углам вычисляются квадратурой Гаусса — Лежандра: 20 узлов по косинусу полярного угла и "
     "2×24 узла по азимуту, что с учётом симметрии даёт 960 лучей на точку. Чтобы избежать потери точности "
     "в оптически толстых полосах, вдоль луча переносится непосредственно разность $B_\\nu(T(r))-I_\\nu$.")

heading("2.2. Приближение Шустера — Шварцшильда")
text("В приближении принимается, что интенсивность изотропна в пределах каждой из двух полусфер [5, 6]. "
     "Тогда полупотоки $F_\\nu^+$ (наружу) и $F_\\nu^-$ (к оси) удовлетворяют уравнениям [7]")
eq_ss1 = equation(r"\frac{1}{r}\frac{d}{dr}\left(rF_\nu^+\right)=2k_\nu\left(\pi B_\nu-F_\nu^+\right)+f_\nu,")
eq_ss2 = equation(r"\frac{1}{r}\frac{d}{dr}\left(rF_\nu^-\right)=2k_\nu\left(F_\nu^--\pi B_\nu\right)+f_\nu")
text("с граничными условиями", indent=False)
eq_bc = equation(r"F_\nu^-(R)=0,\quad F_\nu^+(0)=F_\nu^-(0).")
text(F("Член $f_\\nu$ описывает переход излучения из внутренней полусферы в наружную при прохождении луча мимо "
     "оси. В работе использовано замыкание $f_\\nu=F_\\nu^-/r$. [[Согласовать с руководителем: в методических "
     "материалах этот член задаётся через интенсивность на касательных лучах.]] Тогда уравнения "
     "({})–({}) принимают вид", eq_ss1, eq_ss2))
eq_cl1 = equation(r"\frac{dF_\nu^-}{dr}=2k_\nu\left(F_\nu^--\pi B_\nu\right),")
eq_cl2 = equation(r"\frac{d}{dr}\left(rF_\nu^+\right)=2k_\nu r\left(\pi B_\nu-F_\nu^+\right)+F_\nu^-.")
text("Дивергенция потока вычисляется по формуле", indent=False)
eq_qss = equation(r"q_\nu=4\pi k_\nu B_\nu-2k_\nu\left(F_\nu^++F_\nu^-\right).")
text(F("Уравнения ({})–({}) интегрируются на радиальной сетке с аналитическим решением в пределах ячейки. "
     "В оптически толстых полосах $F_\\nu^\\pm\\approx\\pi B_\\nu$, и прямое вычисление $q_\\nu$ по формуле ({}) "
     "приводит к потере значащих цифр. Поэтому уравнения решаются для дефицитов "
     "$D^\\pm=\\pi B_\\nu(T(r_0))-F_\\nu^\\pm$, а сетка геометрически сгущается к точке наблюдения $r_0$.", eq_cl1, eq_cl2, eq_qss))

heading("2.3. Предельные случаи")
text("В прозрачном пределе ($\\tau_\\nu\\to 0$) оба метода дают")
eq_thin = equation(r"q_\nu=4\pi k_\nu B_\nu.")
text(F("В диффузионном пределе ($\\tau_\\nu\\to\\infty$) точное решение переходит в приближение лучистой "
     "теплопроводности [1, 3, 4], а разложение уравнений ({})–({}) по степеням $1/k_\\nu$ даёт аналогичное "
     "выражение с другим коэффициентом:", eq_cl1, eq_cl2))
eq_diff = equation(r"F_\nu^{\mathrm{точн}}=-\frac{4\pi}{3k_\nu}\frac{dB_\nu}{dr},\quad "
                   r"F_\nu^{\mathrm{ШШ}}=-\frac{\pi}{k_\nu}\frac{dB_\nu}{dr}.")
text("Следовательно, вне оси", indent=False)
eq_34 = equation(r"\frac{q_\nu^{\mathrm{ШШ}}}{q_\nu^{\mathrm{точн}}}\to\frac{3}{4}.")
text("На оси, где $dB_\\nu/dr=0$, точное значение равно $q_\\nu(0)=-(8\\pi/3k_\\nu)\\,B_\\nu^{\\prime\\prime}$, а приближённое — "
     "$-(\\pi/k_\\nu)\\,B_\\nu^{\\prime\\prime}$, откуда")
eq_38 = equation(r"\frac{q_\nu^{\mathrm{ШШ}}(0)}{q_\nu^{\mathrm{точн}}(0)}\to\frac{3}{8}.")
text(F("Дополнительное занижение на оси связано с граничным условием ({}) и принятым замыканием.", eq_bc))

# ---------------------------------------------------------------- 4
heading("3. Проверка численной реализации")
text("Реализация обоих методов проверена на задачах с известным решением.")
text("1. Для однородного цилиндра приближённое решение совпадает с аналитическим с относительной "
     "погрешностью 10⁻¹⁰, в том числе при $k_\\nu R=100$.")
text(F("2. В прозрачном пределе оба метода воспроизводят формулу ({}) с погрешностью менее 0,1 %.", eq_thin))
text("3. Интегральный баланс энергии")
eq_bal = equation(r"\int_0^R Q(r)\,2\pi r\,dr=2\pi R\,F(R)")
text("выполняется с погрешностью 0,5 % для точного метода и 0,4 % для приближённого (21 точка по радиусу).",
     indent=False)
text(F("4. Диффузионные пределы ({}) и ({}) воспроизводятся с погрешностью менее 3 %.", eq_34, eq_38))
text("5. Сгущение сеток вчетверо меняет результаты менее чем на 1 %.")

# ---------------------------------------------------------------- 5
heading("4. Результаты")
text("Расчёты выполнены при [[T₀ = 11 000 К, Tw = 1800 К, R = 1 см, n = 2 — уточнить параметры варианта]].")

heading("4.1. Спектр оптической толщины")
text(F("Радиальная оптическая толщина полос ({}) меняется от 4,9·10⁻⁴ до 2,4·10³, медиана равна 0,022 (рис. 1). "
     "Из 193 полос 134 имеют $\\tau_\\nu<0,1$; 17 — $0,1\\le\\tau_\\nu<1$; 10 — $1\\le\\tau_\\nu<10$; "
     "32 — $\\tau_\\nu\\ge 10$. Наиболее толстые полосы отвечают резонансным линиям ксенона в вакуумном "
     "ультрафиолете.", eq_tau))
figure("fig1_tau.png", "Радиальная оптическая толщина спектральных полос")

heading("4.2. Спектральная дивергенция потока")
text("На рис. 2 приведены спектры $q_\\nu$ на оси цилиндра. В прозрачной части спектра кривые двух методов "
     "совпадают, в области сильных линий приближение даёт заниженные значения.")
figure("fig2_spectrum_axis.png",
       "Спектр дивергенции потока на оси: точный метод и приближение Шустера — Шварцшильда")
text(F("Зависимость отношения $q_\\nu^{\\mathrm{ШШ}}/q_\\nu^{\\mathrm{точн}}$ от оптической толщины полосы "
     "показана на рис. 3, значения на оси сведены в табл. 1. При $\\tau_\\nu\\ge 10$ отношение на оси выходит "
     "на 3/8, а при $r/R=0,5$ — на 3/4 (медиана 0,746), что согласуется с ({}) и ({}). Вблизи поверхности "
     "($r/R\\ge 0,9$) в толстых полосах диффузионное приближение неприменимо, и отношение существенно "
     "отклоняется от предельных значений.", eq_34, eq_38))
table("Отношение $q_\\nu^{\\mathrm{ШШ}}/q_\\nu^{\\mathrm{точн}}$ на оси цилиндра",
      [["Оптическая толщина", "Число полос", "Медиана", "Диапазон"],
       ["$\\tau_\\nu<0,1$", "134", "0,997", "0,967–1,000"],
       ["$0,1\\le\\tau_\\nu<1$", "17", "0,946", "0,834–0,986"],
       ["$1\\le\\tau_\\nu<10$", "10", "0,553", "0,426–0,759"],
       ["$\\tau_\\nu\\ge 10$", "32", "0,377", "0,376–0,438"]],
      [5.0, 3.5, 3.5, 4.5])
figure("fig4_ratio_vs_tau.png",
       "Отношение приближённой дивергенции к точной в зависимости от оптической толщины полосы")

heading("4.3. Интегральные характеристики")
text("Интегральная дивергенция $Q(r)$ показана на рис. 4 и в табл. 2. В центральной зоне $Q>0$ "
     "(на оси 3,05·10⁹ Вт/м³), при $r/R\\approx 0,54$ знак меняется, и периферийные слои поглощают "
     "излучение ядра.")
table("Интегральная по спектру дивергенция потока",
      [["$r/R$", "0", "0,25", "0,5", "0,75", "0,9", "1,0"],
       ["$Q_{\\mathrm{точн}}$, Вт/м³", "3,05·10⁹", "1,61·10⁹", "6,36·10⁷", "−3,33·10⁷", "−4,21·10⁷", "−5,49·10⁷"],
       ["$Q_{\\mathrm{ШШ}}/Q_{\\mathrm{точн}}$", "0,887", "0,972", "1,031", "1,021", "0,938", "0,956"]],
      [3.8, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2])
figure("fig3_integrated.png",
       "Интегральная по спектру дивергенция потока и отношение результатов двух методов")
text("Плотность потока излучения с поверхности составляет 1,13·10⁶ Вт/м² по точному расчёту и 1,09·10⁶ Вт/м² "
     "в приближении (занижение 3,3 %). Вклад групп полос в радиационные потери на оси приведён в табл. 3.")
table("Вклад групп полос в $Q$ на оси цилиндра",
      [["Оптическая толщина", "Доля в $Q_{\\mathrm{точн}}(0)$", "$Q_{\\mathrm{ШШ}}/Q_{\\mathrm{точн}}$ по группе"],
       ["$\\tau_\\nu<0,1$", "62,5 %", "0,996"],
       ["$0,1\\le\\tau_\\nu<1$", "17,2 %", "0,928"],
       ["$1\\le\\tau_\\nu<10$", "13,3 %", "0,585"],
       ["$\\tau_\\nu\\ge 10$", "6,9 %", "0,378"]],
      [5.0, 5.0, 6.0])

# ---------------------------------------------------------------- 6
heading("5. Обсуждение")
text("Погрешность приближения определяется оптической толщиной полосы и положением точки. Интегральная "
     "погрешность 11 % на оси складывается в основном из вклада полос с $\\tau_\\nu\\ge 1$: они дают 20 % "
     "радиационных потерь, и в них приближение занижает дивергенцию в 1,7–2,7 раза. Прозрачные полосы дают "
     "62,5 % потерь и рассчитываются практически точно.")
text("Отсюда следует, что интегральная точность приближения зависит от того, какая доля энергии переносится "
     "в оптически плотной части спектра. С ростом радиуса столба или плотности газа эта доля увеличивается, "
     "и погрешность должна расти. Оценочный расчёт с коэффициентами поглощения, увеличенными в 100 раз, "
     "показал: число полос с $0,1\\le\\tau_\\nu\\le 10$ возрастает с 27 до 131, а погрешность $Q$ у поверхности "
     "достигает 11 %.")
text("В диффузионном пределе вне оси погрешность сводится к постоянному множителю 3/4. Его можно устранить "
     "перенормировкой коэффициента в двухпоточных уравнениях. В промежуточной области $\\tau_\\nu\\sim 1$ "
     "погрешность не сводится к множителю и зависит от радиуса. Этот вывод согласуется с общим свойством "
     "двухпоточных моделей: они точны в прозрачном пределе и дают неверный коэффициент лучистой "
     "теплопроводности в оптически плотной среде [3, 4].")
text("Занижение на оси до 3/8 связано с принятым замыканием члена $f_\\nu$. Сравнение с вариантом, в котором "
     "$f_\\nu$ вычисляется по касательным лучам, а также с адаптивным алгоритмом работы [7] является предметом "
     "дальнейшей работы.")
text("Ограничения работы: один набор параметров $T_0$, $n$, $R$; ступенчатая аппроксимация спектра внутри "
     "полос; заданное, а не самосогласованное поле температур.")

# ---------------------------------------------------------------- 7
heading("Заключение")
text("1. В оптически прозрачных полосах ($\\tau_\\nu<0,1$) приближение Шустера — Шварцшильда воспроизводит "
     "спектральную дивергенцию потока с погрешностью не более нескольких процентов.")
text("2. В оптически толстых полосах отношение приближённой дивергенции к точной стремится к 3/4 вне оси "
     "и к 3/8 на оси. Пределы получены аналитически и подтверждены численно.")
text("3. Для рассмотренного столба ксеноновой плазмы погрешность интегральной дивергенции составила 11 % "
     "на оси и 2–6 % в остальном объёме, погрешность потока с поверхности — 3,3 %.")
text("4. Интегральная погрешность определяется долей энергии, переносимой в полосах с $\\tau_\\nu\\ge 1$, "
     "и возрастает с увеличением оптической плотности столба.")

# ---------------------------------------------------------------- литература
heading("Список литературы")
REFERENCES = [
    "Зельдович Я. Б., Райзер Ю. П. Физика ударных волн и высокотемпературных гидродинамических явлений. "
    "2-е изд. М.: Наука, 1966.",
    "Четверушкин Б. Н. Математическое моделирование задач динамики излучающего газа. М.: Наука, 1985.",
    "Зигель Р., Хауэлл Дж. Теплообмен излучением / пер. с англ. под ред. Б. А. Хрусталева. М.: Мир, 1975.",
    "Modest M. F. Radiative Heat Transfer. 3rd ed. Oxford: Academic Press, 2013.",
    "Schuster A. Radiation through a foggy atmosphere // Astrophysical Journal. 1905. Vol. 21, no. 1. P. 1–22.",
    "Schwarzschild K. Ueber das Gleichgewicht der Sonnenatmosphäre // Nachrichten von der Gesellschaft der "
    "Wissenschaften zu Göttingen. Mathematisch-Physikalische Klasse. 1906. S. 41–53.",
    "Круковский А. Ю., Болдарев А. С., Ольховская О. Г. Модель Шустера — Шварцшильда для цилиндрически "
    "симметричных течений излучающего газа // Препринты ИПМ им. М. В. Келдыша. 2021. № 42. 19 с. "
    "DOI: 10.20948/prepr-2021-42.",
    "Новиков А. А., Ким Д. А., Вичев И. Ю., Соломянная А. Д., Грушин А. С., Ярцев Б. Л. Численное исследование "
    "излучающей плазмы ксенона // Препринты ИПМ им. М. В. Келдыша. 2022. № 57. 15 с. "
    "DOI: 10.20948/prepr-2022-57.",
]
for number, reference in enumerate(REFERENCES, 1):
    par = fmt(doc.add_paragraph(), indent=False, spacing=1.15, after=3)
    par.paragraph_format.left_indent = Cm(0.9)
    par.paragraph_format.first_line_indent = Cm(-0.9)
    rich(par, f"{number}. {reference}", size=12)

# ---------------------------------------------------------------- English block
fmt(doc.add_paragraph(), indent=False, after=6)
centered("ACCURACY OF THE SCHUSTER–SCHWARZSCHILD APPROXIMATION FOR RADIATIVE TRANSFER IN A CYLINDRICAL XENON "
         "PLASMA COLUMN WITH STRONGLY NON-GREY ABSORPTION", size=12, bold=True, after=6)
small("**Abstract.** The accuracy of the two-flux Schuster–Schwarzschild approximation is studied for the spectral "
      "divergence of the radiative flux in an infinite cylinder with a prescribed temperature profile. The medium "
      "is xenon at 1.5 MPa; the absorption coefficient is tabulated in 193 spectral bands whose radial optical "
      "thickness ranges from 5·10⁻⁴ to 2.4·10³. Direct ray-by-ray integration of the transfer equation is used "
      "as the reference. The error is governed by the band optical thickness: in thin bands (τ < 0.1) it does "
      "not exceed a few per cent, while in thick bands the ratio of the approximate to the exact divergence "
      "tends to 3/4 off the axis and to 3/8 on the axis. Both limits are derived analytically and confirmed "
      "numerically. The error of the spectrally integrated divergence is 11 % on the axis and 2–6 % elsewhere; "
      "the surface flux is underestimated by 3.3 %.", after=6)
small("**Keywords:** radiative transfer, Schuster–Schwarzschild approximation, two-flux approximation, radiative "
      "flux divergence, optical thickness, diffusion limit, xenon plasma.")

out = REVIEW / "ARTICLE.docx"
doc.save(out)
print("saved", out, "| формул:", counters["eq"], "рисунков:", counters["fig"], "таблиц:", counters["tab"])
