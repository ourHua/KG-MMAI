"""
labels.py — renders every entity name in Latin script.

Policy
------
Herbs (HER) and prescriptions (PRE) take pinyin, which is the convention in the
English-language TCM literature and keeps the identifier searchable.
Symptoms (SYM), causes (CAU), and effects (EFF) take an English gloss where one
is established, since a transliterated symptom is not informative to a reader.
Anything without a curated gloss falls back to pinyin, so the function is total
over the 8,024-entity vocabulary and no figure can silently emit CJK.

`export_table()` writes entity_labels.csv, the full mapping, which ships with
the experiment package so that any label in the paper can be traced back to the
source term.
"""

__author__ = "LIJUNHUA"

try:
    from pypinyin import lazy_pinyin
except ImportError:  # The curated labels still work without the optional package.
    lazy_pinyin = None

# --------------------------------------------------------------------------- #
# curated glosses — terminology follows the WHO International Standard
# Terminologies on Traditional Medicine in the Western Pacific Region
# --------------------------------------------------------------------------- #
GLOSS = {
    # ---- pulse findings (annotated CAU in this corpus) ----
    "脉弦": "Wiry pulse",
    "脉细数": "Fine rapid pulse",
    "脉沉细": "Deep fine pulse",
    "脉滑数": "Slippery rapid pulse",
    "脉弦数": "Wiry rapid pulse",
    "脉细": "Fine pulse",
    "脉沉": "Deep pulse",
    "脉数": "Rapid pulse",
    "脉浮": "Floating pulse",
    "脉乍紧者": "Abruptly tight pulse",
    "六脉弦紧": "Six pulses wiry and tight",
    # ---- pathogenesis ----
    "气虚": "Qi deficiency",
    "风痰": "Wind-phlegm",
    "肺虚": "Lung deficiency",
    "脾弱": "Spleen weakness",
    "脾失健运": "Spleen failing to transport",
    "火烁肺金": "Fire scorching lung metal",
    "伤寒": "Cold damage",
    # ---- symptoms and signs ----
    "发热": "Fever",
    "舌红": "Red tongue",
    "舌质红": "Red tongue body",
    "舌淡": "Pale tongue",
    "苔白": "White tongue coating",
    "苔薄白": "Thin white coating",
    "苔黄": "Yellow tongue coating",
    "头痛": "Headache",
    "口渴": "Thirst",
    "腹痛": "Abdominal pain",
    "腹胀": "Abdominal distension",
    "盗汗": "Night sweating",
    "肢冷": "Cold limbs",
    "便溏": "Loose stool",
    "大便秘结": "Constipation",
    "胸闷": "Chest oppression",
    "恶心呕吐": "Nausea and vomiting",
    "咳嗽": "Cough",
    "有汗": "Sweating",
    "喉燥": "Dry throat",
    "手足厥冷": "Cold reversal of extremities",
    "痰浊内生": "Internal phlegm-turbidity",
    "痰浊中阻": "Phlegm-turbidity obstructing centre",
    "红斑丘疹": "Erythematous papules",
    "皮肤紫斑多": "Purpuric skin macules",
    "红肿烂斑大痛": "Red swollen ulcerated macules",
    "黑睛猝然偏斜": "Sudden deviation of the dark of the eye",
    # ---- treatment effects ----
    "清热": "Clear heat",
    "健脾": "Fortify the spleen",
    "益气": "Boost qi",
    "补气": "Supplement qi",
    "补肾": "Supplement the kidney",
    "滋阴": "Enrich yin",
    "止血": "Stanch bleeding",
    "活血化瘀": "Activate blood, resolve stasis",
    "清热解毒": "Clear heat, resolve toxin",
    "健脾益气": "Fortify spleen, boost qi",
}

# herbs and prescriptions: pinyin, with a familiar equivalent where one exists
PINYIN_EXTRA = {
    "甘草": "Gancao",
    "茯苓": "Fuling",
    "当归": "Danggui",
    "白术": "Baizhu",
    "黄芩": "Huangqin",
    "柴胡": "Chaihu",
    "白芍": "Baishao",
    "黄连": "Huanglian",
    "黄芪": "Huangqi",
    "人参": "Renshen",
    "陈皮": "Chenpi",
    "厚朴": "Houpo",
    "大黄": "Dahuang",
    "栀子": "Zhizi",
    "连翘": "Lianqiao",
    "桃仁": "Taoren",
    "红花": "Honghua",
    "枳壳": "Zhiqiao",
    "苍术": "Cangzhu",
    "五味": "Wuwei",
    "补中益气汤": "Buzhong Yiqi Tang",
    "六君子汤": "Liujunzi Tang",
    "二陈汤": "Erchen Tang",
    "四物汤": "Siwu Tang",
    "五苓散": "Wuling San",
    "逍遥散": "Xiaoyao San",
    "理中汤": "Lizhong Tang",
    "龙胆泻肝汤": "Longdan Xiegan Tang",
    "八珍汤": "Bazhen Tang",
    "血府逐瘀汤": "Xuefu Zhuyu Tang",
    "异功散": "Yigong San",
    "三黄洗剂": "Sanhuang Xiji",
    "五味消毒饮": "Wuwei Xiaodu Yin",
    "参苓白术散": "Shenling Baizhu San",
    "桂枝汤": "Guizhi Tang",
    "犀角地黄汤": "Xijiao Dihuang Tang",
    "玄明粉": "Xuanmingfen",
    "续命汤": "Xuming Tang",
}

_SUFFIX = {"汤": " Tang", "散": " San", "丸": " Wan", "饮": " Yin", "膏": " Gao"}


def pinyin(name: str) -> str:
    """Return a Latin-script label, using pinyin when available."""
    if name in PINYIN_EXTRA:
        return PINYIN_EXTRA[name]
    if lazy_pinyin is None:
        # Keep the fallback ASCII-only and reversible when pypinyin is absent.
        return "U" + "-".join(f"{ord(ch):04X}" for ch in name)
    for cjk, suffix in _SUFFIX.items():
        if name.endswith(cjk) and len(name) > 1:
            stem = "".join(lazy_pinyin(name[:-1]))
            return stem.capitalize() + suffix
    syllables = lazy_pinyin(name)
    return "".join(syllables).capitalize() if syllables else name


def label(name: str, etype: str = "") -> str:
    """Latin-script label for an entity. Total: never returns CJK."""
    if etype in ("HER", "PRE"):
        return pinyin(name)
    return GLOSS.get(name) or pinyin(name)


def short(name: str, etype: str = "", width: int = 26) -> str:
    """Label truncated for use as an axis tick."""
    s = label(name, etype)
    return s if len(s) <= width else s[: width - 1].rstrip() + "\u2026"


def has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(s))


def export_table(nodes_df, path):
    """Write the full entity -> label mapping shipped with the package."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "source_term", "type", "pinyin", "label", "curated_gloss"])
        for r in nodes_df.itertuples(index=False):
            w.writerow([r.id, r.name, r.type, pinyin(r.name),
                        label(r.name, r.type), r.name in GLOSS])
    return path
