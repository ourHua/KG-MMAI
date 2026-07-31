"""
labels_en.py — English rendering of TCM entity names for figures and text.

Convention follows standard practice in the English-language TCM literature:

  * Materia medica (HER) and formulas (PRE) are given in hyphenated pinyin,
    which is the accepted citation form and is reversible to the original term.
  * Clinical terms — symptoms (SYM), pulse qualities and pathogenesis (CAU),
    and treatment actions (EFF) — are given as English glosses, since a pinyin
    rendering of these carries no information for a non-Chinese reader.

Any entity not in the curated table falls back to hyphenated pinyin via
`pypinyin`, and to the original string if that package is unavailable. The
mapping is exported to results/entity_labels_en.csv so that every label used
in a figure can be traced back to its source term.
"""
from __future__ import annotations

__author__ = "LIJUNHUA"

try:
    from pypinyin import lazy_pinyin
    _HAVE_PINYIN = True
except ImportError:                                     # pragma: no cover
    _HAVE_PINYIN = False

# --------------------------------------------------------------------------- #
# Curated glosses. Covers every entity that appears in a figure or in the text,
# plus the most frequent members of each type.
# --------------------------------------------------------------------------- #
GLOSS = {
    # ---- CAU: pulse qualities and pathogenesis --------------------------- #
    "脉细数": "thready-rapid pulse",
    "脉沉细": "deep-thready pulse",
    "脉弦": "wiry pulse",
    "脉滑数": "slippery-rapid pulse",
    "脉细": "thready pulse",
    "脉弦数": "wiry-rapid pulse",
    "脉濡": "soggy pulse",
    "脉浮": "floating pulse",
    "脉沉": "deep pulse",
    "脉数": "rapid pulse",
    "脉紧": "tight pulse",
    "脉迟": "slow pulse",
    "脉滑": "slippery pulse",
    "六脉弦紧": "wiry-tight pulse (all six)",
    "脉乍紧者": "abruptly tight pulse",
    "气虚": "qi deficiency",
    "风痰": "wind-phlegm",
    "肺虚": "lung deficiency",
    "伤寒": "cold damage",
    "脾失健运": "spleen failing to transport",
    "脾弱": "spleen weakness",
    "火烁肺金": "fire scorching lung metal",
    "血虚": "blood deficiency",
    "阴虚": "yin deficiency",
    "阳虚": "yang deficiency",
    "湿热": "damp-heat",
    "肝郁": "liver constraint",

    # ---- HER: materia medica (pinyin) ------------------------------------ #
    "甘草": "Gan-cao",
    "茯苓": "Fu-ling",
    "当归": "Dang-gui",
    "黄芩": "Huang-qin",
    "白术": "Bai-zhu",
    "陈皮": "Chen-pi",
    "黄芪": "Huang-qi",
    "黄连": "Huang-lian",
    "柴胡": "Chai-hu",
    "白芍": "Bai-shao",
    "人参": "Ren-shen",
    "大黄": "Da-huang",
    "五味": "Wu-wei",
    "厚朴": "Hou-po",
    "连翘": "Lian-qiao",
    "栀子": "Zhi-zi",
    "桃仁": "Tao-ren",
    "红花": "Hong-hua",
    "枳壳": "Zhi-qiao",
    "半夏": "Ban-xia",
    "生地": "Sheng-di",
    "熟地": "Shu-di",
    "川芎": "Chuan-xiong",
    "泽泻": "Ze-xie",

    # ---- PRE: formulas (pinyin) ------------------------------------------ #
    "补中益气汤": "Buzhong-yiqi-tang",
    "六君子汤": "Liujunzi-tang",
    "二陈汤": "Erchen-tang",
    "四物汤": "Siwu-tang",
    "五苓散": "Wuling-san",
    "逍遥散": "Xiaoyao-san",
    "理中汤": "Lizhong-tang",
    "龙胆泻肝汤": "Longdan-xiegan-tang",
    "八珍汤": "Bazhen-tang",
    "异功散": "Yigong-san",
    "血府逐瘀汤": "Xuefu-zhuyu-tang",
    "参苓白术散": "Shenling-baizhu-san",
    "桂枝汤": "Guizhi-tang",
    "犀角地黄汤": "Xijiao-dihuang-tang",
    "五味消毒饮": "Wuwei-xiaodu-yin",
    "三黄洗剂": "Sanhuang-xiji",
    "玄明粉": "Xuanming-fen",
    "续命汤": "Xuming-tang",
    "苍术": "Cang-zhu",          # labelled PRE in the corpus; see Section 5.4

    # ---- EFF: treatment actions ------------------------------------------ #
    "清热": "clear heat",
    "健脾": "fortify spleen",
    "益气": "boost qi",
    "补肾": "tonify kidney",
    "活血化瘀": "invigorate blood",
    "清热解毒": "clear heat, resolve toxin",
    "滋阴": "nourish yin",
    "补气": "tonify qi",
    "止血": "stanch bleeding",
    "健脾益气": "fortify spleen, boost qi",
    "祛湿": "dispel dampness",
    "疏肝": "course the liver",
    "养血": "nourish blood",

    # ---- SYM: symptoms and signs ----------------------------------------- #
    "发热": "fever",
    "舌红": "red tongue",
    "头痛": "headache",
    "口渴": "thirst",
    "腹痛": "abdominal pain",
    "舌质红": "red tongue body",
    "舌淡": "pale tongue",
    "苔白": "white coating",
    "苔薄白": "thin white coating",
    "苔黄": "yellow coating",
    "盗汗": "night sweats",
    "肢冷": "cold limbs",
    "便溏": "loose stool",
    "腹胀": "abdominal distension",
    "大便秘结": "constipation",
    "胸闷": "chest oppression",
    "恶心呕吐": "nausea and vomiting",
    "咳嗽": "cough",
    "有汗": "sweating present",
    "喉燥": "dry throat",
    "手足厥冷": "cold reversal of extremities",
    "黑睛猝然偏斜": "sudden deviation of the dark of the eye",
    "红肿烂斑大痛": "red swollen ulcerated patches, severe pain",
    "皮肤紫斑多": "purpuric skin patches",
    "红斑丘疹": "erythematous papules",
    "痰浊中阻": "turbid phlegm obstructing the centre",
    "痰浊内生": "turbid phlegm generated internally",
    "自汗": "spontaneous sweating",
    "乏力": "fatigue",
    "纳呆": "poor appetite",
}


def to_pinyin(name: str) -> str:
    if not _HAVE_PINYIN:
        return name
    parts = lazy_pinyin(name)
    return "-".join(p.capitalize() if i == 0 else p
                    for i, p in enumerate(parts))


def label(name: str, etype: str | None = None) -> str:
    """English label for one entity name."""
    if name in GLOSS:
        return GLOSS[name]
    if name.isascii():
        return name
    return to_pinyin(name)


def label_series(names, types=None):
    if types is None:
        return [label(n) for n in names]
    return [label(n, t) for n, t in zip(names, types)]


def export_mapping(names, types, path):
    """Write the name -> label mapping actually used, for traceability."""
    import csv
    rows = sorted({(n, t, label(n, t),
                    "curated" if n in GLOSS else "pinyin")
                   for n, t in zip(names, types)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_name", "type", "label_en", "provenance"])
        w.writerows(rows)
    return len(rows)
