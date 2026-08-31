"""HunyuanOCR 专用字段与极简 Prompt；不读取通用抽取配置。"""

from __future__ import annotations


ID_CARD_FRONT_FIELDS = (
    "证件类型", "姓名", "身份证号", "出生日期", "性别", "民族", "住址",
)
ID_CARD_BACK_FIELDS = ("签发机关", "有效期")

HUNYUAN_DOCUMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "driver_license": (
        "证号", "姓名", "性别", "国籍", "住址", "出生日期",
        "初次领证日期", "准驾车型", "有效期限", "档案编号", "证件标题",
    ),
    "vehicle_license": (
        "证件类型", "号牌号码", "车辆类型", "所有人", "住址", "使用性质", "品牌型号",
        "车辆识别代号", "发动机号码", "注册日期", "发证日期", "档案编号",
        "核定载人数", "总质量", "整备质量", "外廓尺寸", "检验记录",
    ),
    "registration_certificate": (
        "登记证编号", "机动车所有人", "身份证明名称", "号码", "登记机关",
        "登记日期", "机动车登记编号", "车辆类型", "车辆品牌", "车辆型号",
        "车身颜色", "车架号", "国产/进口", "发动机号", "发动机型号",
        "燃料类型", "排量", "功率", "制造厂名称", "转向形式", "前轮",
        "后轮", "轮胎数", "轮胎规格", "钢板弹簧片数", "轴距", "轴数",
        "外廓尺寸", "内部尺寸", "总质量", "核定载客", "核定载质量",
        "驾驶室载客", "准牵引总质量", "车辆获得方式", "使用性质",
        "出厂日期", "发证日期",
    ),
}

# 混合复核输入的是单个字段的局部裁剪，不再使用整证字段抽取 Prompt。
LOCAL_RECHECK_PROMPT = "识别图片中的文字。"


def _prompt(fields: tuple[str, ...]) -> str:
    field_list = ", ".join(repr(field) for field in fields)
    return f"提取图片中的: [{field_list}] 的字段内容，并按照JSON格式返回。"


def document_prompt(document_type: str) -> str:
    """获取驾驶证、行驶证或登记证的 HunyuanOCR 专用 Prompt。"""

    try:
        fields = HUNYUAN_DOCUMENT_FIELDS[document_type]
    except KeyError as exc:
        raise ValueError(f"未配置 HunyuanOCR 证件 Prompt：{document_type}") from exc
    return _prompt(fields)


def id_card_prompt(side: str = "") -> str:
    """获取身份证正面、反面或完整字段的 HunyuanOCR 专用 Prompt。"""

    fields = (
        ID_CARD_FRONT_FIELDS
        if side == "front"
        else ID_CARD_BACK_FIELDS
        if side == "back"
        else ID_CARD_FRONT_FIELDS + ID_CARD_BACK_FIELDS
    )
    return _prompt(fields)


def local_recheck_prompt() -> str:
    """获取混合模型局部裁剪复核的纯 OCR Prompt。"""

    return LOCAL_RECHECK_PROMPT
