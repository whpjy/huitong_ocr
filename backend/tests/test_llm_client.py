from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest

from llm_manager.client import OpenAICompatibleClient, parse_json_object
from llm_manager.config import LLMProviderConfig
from llm_manager.hunyuan_ocr_adapter import (
    contains_spotting_coordinates,
    merge_document_pages,
    merge_id_card_pages,
    normalize_document_page,
    spotting_markdown,
)
from llm_manager.hunyuan_ocr_prompts import document_prompt, id_card_prompt
from llm_manager.profiles import load_profiles


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        content: str = "",
        text: str = "",
        stream_lines: list[str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._content = content
        self.text = text
        self.stream_lines = stream_lines or []
        self.closed = False

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": self._content,
                    }
                }
            ]
        }

    def iter_lines(self, *, decode_unicode: bool = False):
        assert decode_unicode is True
        yield from self.stream_lines

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def provider_config(
    *,
    max_attempts: int = 1,
    input_mode: str = "text",
    vision_content_order: str = "images_first",
) -> LLMProviderConfig:
    return LLMProviderConfig(
        name="test_model",
        type="openai_compatible",
        enabled=True,
        base_url="https://example.test/v1",
        endpoint="/chat/completions",
        api_key="test-secret",
        api_key_env="",
        model="test-model",
        timeout=9,
        max_attempts=max_attempts,
        # Canned FakeSession responses are positional. Tests that exercise
        # concurrent image calls opt in to concurrency=2 explicitly.
        concurrency=1,
        temperature=0.0,
        response_format="json_object",
        extra_body={"enable_thinking": True},
        input_mode=input_mode,
        vision_content_order=vision_content_order,
    )


def test_hunyuan_prompts_are_dedicated_and_minimal() -> None:
    driver_prompt = document_prompt("driver_license")
    vehicle_prompt = document_prompt("vehicle_license")
    front_prompt = id_card_prompt("front")
    back_prompt = id_card_prompt("back")

    assert driver_prompt == (
        "提取图片中的: ['证号', '姓名', '性别', '国籍', '住址', '出生日期', "
        "'初次领证日期', '准驾车型', '有效期限', '档案编号', '证件标题'] "
        "的字段内容，并按照JSON格式返回。"
    )
    assert front_prompt == (
        "提取图片中的: ['证件类型', '姓名', '身份证号', '出生日期', '性别', "
        "'民族', '住址'] 的字段内容，并按照JSON格式返回。"
    )
    assert back_prompt == (
        "提取图片中的: ['签发机关', '有效期'] 的字段内容，并按照JSON格式返回。"
    )
    assert vehicle_prompt == (
        "提取图片中的: ['证件类型', '号牌号码', '车辆类型', '所有人', '住址', '使用性质', "
        "'品牌型号', '车辆识别代号', '发动机号码', '注册日期', '发证日期', "
        "'档案编号', '核定载人数', '总质量', '整备质量', '外廓尺寸', "
        "'检验记录'] 的字段内容，并按照JSON格式返回。"
    )
    assert "字段规则" not in driver_prompt
    assert "OCR 文本" not in driver_prompt
    assert "证件类型" not in driver_prompt
    assert "证件标题" in driver_prompt
    assert "证件类型" in vehicle_prompt
    assert "核定载质量" not in vehicle_prompt


def test_batch_prompt_key_is_used_in_hunyuan_document_payload(tmp_path) -> None:
    image_path = tmp_path / "driver.jpg"
    image_path.write_bytes(b"fake-image")
    client = OpenAICompatibleClient(
        provider_config(input_mode="vision"),
        session_factory=lambda: FakeSession([]),
    )

    payload = client.build_hunyuan_document_payload(
        load_profiles().get("driver_license"),
        image_path,
        prompt_key="driver_license",
    )

    content = payload["messages"][1]["content"]
    text_part = next(item for item in content if item["type"] == "text")
    assert text_part["text"] == document_prompt("driver_license")


def test_parse_json_object_accepts_fence_and_surrounding_text() -> None:
    assert parse_json_object('```json\n{"姓名": "张三"}\n```') == {
        "姓名": "张三"
    }
    assert parse_json_object('结果如下：{"姓名": "张三"}。') == {
        "姓名": "张三"
    }


def test_parse_json_object_conservatively_repairs_truncated_final_string() -> None:
    truncated = '{"号牌号码":"新RF2B67","档案编号":"6540022412089X'

    with pytest.raises(json.JSONDecodeError):
        parse_json_object(truncated)

    assert parse_json_object(
        truncated,
        repair_truncated_final_string=True,
    ) == {
        "号牌号码": "新RF2B67",
        "档案编号": "6540022412089X",
    }
    assert parse_json_object(
        '{"备注":"值中包含}但仍被EOS截断',
        repair_truncated_final_string=True,
    ) == {"备注": "值中包含}但仍被EOS截断"}


@pytest.mark.parametrize(
    "invalid",
    [
        '{"档案编号":"6540022412089X"',
        '{"档案编号":',
        'not json',
        '["6540022412089X',
    ],
)
def test_truncated_final_string_repair_rejects_other_damage(invalid: str) -> None:
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_json_object(
            invalid,
            repair_truncated_final_string=True,
        )


def test_profile_normalizes_multiple_records() -> None:
    profile = load_profiles().get("id_card")
    records = profile.normalize_records(
        {
            "records": [
                {"姓名": "张三", "身份证号": "110101199001011237"},
                {"姓名": "李四", "身份证号": "110101199202022346"},
            ]
        }
    )

    assert [record["姓名"] for record in records] == ["张三", "李四"]
    assert records[0]["身份证号"] == "110101199001011237"


def test_extract_records_accepts_top_level_json_array() -> None:
    profile = load_profiles().get("vehicle_license")
    session = FakeSession(
        [
            FakeResponse(
                content='[{"车牌号码":"甘J87W73"},{"车牌号码":"甘A12345"}]'
            )
        ]
    )
    client = OpenAICompatibleClient(
        provider_config(),
        session_factory=lambda: session,
    )

    records = client.extract_records(profile, "行驶证 OCR")

    assert [record["车牌号码"] for record in records] == [
        "甘J87W73",
        "甘A12345",
    ]


def test_extract_image_records_accepts_top_level_json_array(tmp_path) -> None:
    profile = load_profiles().get("vehicle_license")
    image_path = tmp_path / "vehicle.jpg"
    image_path.write_bytes(b"fake-image")
    session = FakeSession(
        [FakeResponse(content='[{"车牌号码":"甘J87W73"}]')]
    )
    client = OpenAICompatibleClient(
        provider_config(input_mode="vision"),
        session_factory=lambda: session,
    )

    records = client.extract_image_records(profile, [image_path])

    assert records[0]["车牌号码"] == "甘J87W73"


def test_client_sends_openai_compatible_payload_and_normalizes() -> None:
    profile = load_profiles().get("id_card")
    response_content = json.dumps(
        {
            "姓名": " 张 三 ",
            "身份证号": "110101199001011237",
        },
        ensure_ascii=False,
    )
    session = FakeSession([FakeResponse(content=response_content)])
    client = OpenAICompatibleClient(
        provider_config(),
        session_factory=lambda: session,
    )

    result = client.extract(profile, "身份证 OCR")

    assert result["姓名"] == "张三"
    assert result["身份证号"] == "110101199001011237"
    call = session.calls[0]
    assert call["url"] == "https://example.test/v1/chat/completions"
    assert call["timeout"] == 9
    assert call["headers"]["Authorization"] == "Bearer test-secret"
    assert call["json"]["model"] == "test-model"
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert call["json"]["enable_thinking"] is True
    assert call["stream"] is False


def test_vision_client_embeds_images_in_multimodal_message(tmp_path) -> None:
    profile = load_profiles().get("id_card")
    image_path = tmp_path / "card.png"
    image_path.write_bytes(b"fake-image")
    response_content = json.dumps({"姓名": "张三"}, ensure_ascii=False)
    session = FakeSession([FakeResponse(content=response_content)])
    client = OpenAICompatibleClient(
        provider_config(input_mode="vision"),
        session_factory=lambda: session,
    )

    result = client.extract_images(profile, [image_path])

    assert result["姓名"] == "张三"
    content = session.calls[0]["json"]["messages"][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert content[-1]["type"] == "text"


def test_vision_client_can_put_text_before_images(tmp_path) -> None:
    profile = load_profiles().get("id_card")
    image_path = tmp_path / "card.png"
    image_path.write_bytes(b"fake-image")
    session = FakeSession([FakeResponse(content='{"records":[{}]}')])
    client = OpenAICompatibleClient(
        provider_config(
            input_mode="vision",
            vision_content_order="text_first",
        ),
        session_factory=lambda: session,
    )

    client.extract_images(profile, [image_path])

    content = session.calls[0]["json"]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_vision_client_can_return_free_form_image_text(tmp_path) -> None:
    image_path = tmp_path / "general.png"
    image_path.write_bytes(b"fake-image")
    session = FakeSession([FakeResponse(content="第一行\n第二行")])
    client = OpenAICompatibleClient(
        provider_config(input_mode="vision"),
        session_factory=lambda: session,
    )

    text = client.recognize_images(
        [image_path],
        system_prompt="忠实识别图片",
        user_prompt="输出全部原始文本",
    )

    assert text == "第一行\n第二行"
    payload = session.calls[0]["json"]
    assert "response_format" not in payload
    assert payload["messages"][0]["content"] == "忠实识别图片"


def test_hunyuan_general_spotting_output_becomes_markdown(tmp_path) -> None:
    image_path = tmp_path / "vehicle.jpg"
    image_path.write_bytes(b"fake-image")
    content = str([
        {"box": [10, 10, 80, 30], "text": "档案编号"},
        {"box": [100, 10, 220, 30], "text": "650100742811"},
        {"box": [10, 40, 80, 60], "text": "总质量"},
        {"box": [100, 40, 180, 60], "text": "2245kg"},
    ])
    session = FakeSession([FakeResponse(content=content)])
    config = replace(
        provider_config(input_mode="vision"),
        vision_response_adapter="hunyuan_ocr",
        response_format=None,
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    text = client.recognize_images(
        [image_path],
        system_prompt="忠实识别",
        user_prompt="使用 Markdown 表格",
    )

    assert contains_spotting_coordinates(content)
    assert "| 字段 | 内容 |" in text
    assert "| 档案编号 | 650100742811 |" in text
    assert "| 总质量 | 2245kg |" in text
    assert "'box'" not in text


def test_spotting_markdown_escapes_table_pipes() -> None:
    content = [{"box": [1, 1, 2, 2], "text": "A|B"}]

    assert "A\\|B" in spotting_markdown(content)


def test_labeled_text_vision_adapter_returns_driver_fields(tmp_path) -> None:
    image_path = tmp_path / "driver.jpg"
    image_path.write_bytes(b"fake-image")
    content = """中华人民共和国机动车驾驶证
证号 450702199602096616
姓名 黄文慧 性别 男 国籍 中国/CHN
Address 广西钦州市钦南区久隆镇青草村委那珠村1号
出生日期 1996-02-09
初次领证日期 2016-03-07
准驾车型 C1
有效期限 2022-03-07 至 2032-03-07
中华人民共和国机动车驾驶证副页
档案编号 450760519433
"""
    session = FakeSession([FakeResponse(content=content)])
    config = replace(
        provider_config(
            input_mode="vision",
            vision_content_order="text_first",
        ),
        vision_response_adapter="labeled_text",
        response_format=None,
        extra_body={"max_tokens": 2048},
    )
    client = OpenAICompatibleClient(
        config,
        session_factory=lambda: session,
    )

    records = client.extract_image_records(
        load_profiles().get("driver_license"),
        [image_path],
    )

    assert records[0]["姓名"] == "黄文慧"
    assert records[0]["证号"] == "450702199602096616"
    assert records[0]["准驾车型"] == "C1"
    assert records[0]["有效期限"] == "20220307-20320307"
    assert records[0]["档案编号"] == "450760519433"
    assert len(session.calls) == 1
    payload = session.calls[0]["json"]
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert payload["max_tokens"] == 2048
    assert "response_format" not in payload


def test_labeled_text_adapter_accepts_split_id_card_labels(tmp_path) -> None:
    image_path = tmp_path / "id-card.jpg"
    image_path.write_bytes(b"fake-image")
    content = """中华人民共和国居民身份证
姓名 袁奉翔
性别 男 民族 汉
出生 1981年10月4日
住址 广东省中山市古镇镇教昌外闸北三巷1号
公民身份号码 442000198110045737
签发机
中山市公安局
有效期限
2015.10.27-2035.10.27
"""
    session = FakeSession([FakeResponse(content=content)])
    config = replace(
        provider_config(input_mode="vision"),
        vision_response_adapter="labeled_text",
        response_format=None,
        extra_body={},
    )
    client = OpenAICompatibleClient(
        config,
        session_factory=lambda: session,
    )

    record = client.extract_images(
        load_profiles().get("id_card"),
        [image_path],
    )

    assert record["姓名"] == "袁奉翔"
    assert record["身份证号"] == "442000198110045737"
    assert record["签发机关"] == "中山市公安局"
    assert record["有效期"] == "20151027-20351027"


def test_hunyuan_adapter_extracts_each_id_card_side_as_json_and_merges(
    tmp_path,
) -> None:
    front = tmp_path / "DG12" / "front.jpg"
    back = tmp_path / "DG13" / "back.jpg"
    front.parent.mkdir()
    back.parent.mkdir()
    front.write_bytes(b"front")
    back.write_bytes(b"back")
    session = FakeSession(
        [
            FakeResponse(
                content="""```json
{"\u8bc1\u4ef6\u7c7b\u578b":"\u8eab\u4efd\u8bc1","\u59d3\u540d":"\u8881\u5949\u7fd4","\u8eab\u4efd\u8bc1\u53f7":"442000198110045737","\u51fa\u751f\u65e5\u671f":"1981\u5e7410\u67084\u65e5","\u6027\u522b":"\u7537","\u6c11\u65cf":"\u6c49","\u4f4f\u5740":"\u5e7f\u4e1c\u7701\u4e2d\u5c71\u5e02\u53e4\u9547\u9547\u6559\u660c\u5916\u95f8\u5317\u4e09\u5df71\u53f7","\u7b7e\u53d1\u673a\u5173":"\u9519误的街道办事处","\u6709\u6548\u671f":"2000.01.01-2010.01.01"}
```"""
            ),
            FakeResponse(
                content="""```json
{"\u8bc1\u4ef6\u7c7b\u578b":"\u5c45\u6c11\u8eab\u4efd\u8bc1","\u59d3\u540d":"\u9519误姓名","\u8eab\u4efd\u8bc1号":"110101199001011237","\u51fa\u751f\u65e5\u671f":"1990.01.01","\u6027\u522b":"\u5973","\u6c11\u65cf":"\u6c49","\u4f4f\u5740":"\u9519误住址","\u7b7e\u53d1\u673a\u5173":"\u4e2d\u5c71\u5e02\u516c\u5b89\u5c40","\u6709\u6548\u671f":"2015.10.27-2035.10.27"}
```"""
            ),
        ]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        # This FakeSession assigns canned responses by call order. Keep this
        # merge-focused test sequential; the following test covers concurrency.
        concurrency=1,
        extra_body={"max_tokens": 2048},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    records = client.extract_image_records(
        load_profiles().get("id_card"),
        [front, back],
    )

    assert records == [
        {
            "证件类型": "第二代身份证",
            "姓名": "袁奉翔",
            "身份证号": "442000198110045737",
            "出生日期": "19811004",
            "性别": "男",
            "民族": "汉",
            "住址": "广东省中山市古镇镇教昌外闸北三巷1号",
            "签发机关": "中山市公安局",
            "有效期": "20151027-20351027",
        }
    ]
    assert len(session.calls) == 2
    for index, call in enumerate(session.calls):
        payload = call["json"]
        assert "response_format" not in payload
        assert payload["messages"][0] == {"role": "system", "content": ""}
        content = payload["messages"][1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"
        assert "并按照JSON格式返回" in content[1]["text"]
        if index == 0:
            assert "姓名" in content[1]["text"]
            assert "签发机关" not in content[1]["text"]
        else:
            assert "签发机关" in content[1]["text"]
            assert "姓名" not in content[1]["text"]
            assert "证件类型" not in content[1]["text"]


def test_hunyuan_adapter_recognizes_id_card_sides_concurrently_then_merges(
    tmp_path,
) -> None:
    front = tmp_path / "DG12" / "front.jpg"
    back = tmp_path / "DG13" / "back.jpg"
    front.parent.mkdir()
    back.parent.mkdir()
    front.write_bytes(b"front")
    back.write_bytes(b"back")

    class ConcurrentSideSession:
        def __init__(self) -> None:
            self.barrier = threading.Barrier(2)
            self.calls: list[str] = []
            self.lock = threading.Lock()

        def post(self, _url: str, **kwargs: object) -> FakeResponse:
            payload = kwargs["json"]
            assert isinstance(payload, dict)
            messages = payload["messages"]
            assert isinstance(messages, list)
            content = messages[1]["content"]
            prompt = content[1]["text"]
            side = "front" if "姓名" in prompt else "back"
            with self.lock:
                self.calls.append(side)
            self.barrier.wait(timeout=1)
            if side == "front":
                return FakeResponse(
                    content=(
                        '{"证件类型":"身份证",'
                        '"姓名":"张三",'
                        '"身份证号":"110101199001011237"}'
                    )
                )
            return FakeResponse(
                content=(
                    '{"签发机关":"北京市公安局",'
                    '"有效期":"2020.01.01-2030.01.01"}'
                )
            )

    session = ConcurrentSideSession()
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        concurrency=2,
        extra_body={"max_tokens": 2048},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    records = client.extract_image_records(
        load_profiles().get("id_card"),
        [front, back],
    )

    assert sorted(session.calls) == ["back", "front"]
    assert records[0]["姓名"] == "张三"
    assert records[0]["身份证号"] == "110101199001011237"
    assert records[0]["签发机关"] == "北京市公安局"
    assert records[0]["有效期"] == "20200101-20300101"


def test_hunyuan_adapter_extracts_single_id_card_without_side_folder(
    tmp_path,
) -> None:
    image_path = tmp_path / "single-card.jpg"
    image_path.write_bytes(b"single-card")
    session = FakeSession(
        [
            FakeResponse(
                content=(
                    '{"证件类型":"居民身份证","姓名":"袁奉翔",'
                    '"身份证号":"442000198110045737","性别":"男",'
                    '"民族":"汉","住址":"广东省中山市",'
                    '"签发机关":"中山市公安局",'
                    '"有效期":"2015.10.27-2035.10.27"}'
                )
            )
        ]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        extra_body={"max_tokens": 2048},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(
        load_profiles().get("id_card"),
        [image_path],
    )

    assert record["姓名"] == "袁奉翔"
    assert record["身份证号"] == "442000198110045737"
    assert record["签发机关"] == "中山市公安局"
    assert record["有效期"] == "20151027-20351027"
    prompt = session.calls[0]["json"]["messages"][1]["content"][1]["text"]
    assert "姓名" in prompt
    assert "签发机关" in prompt


def test_hunyuan_adapter_retries_spotting_and_extracts_driver_json(
    tmp_path,
) -> None:
    image_path = tmp_path / "driver.jpg"
    image_path.write_bytes(b"driver")
    session = FakeSession(
        [
            FakeResponse(
                content='[{"box":[1,2,3,4],"text":"无关页面"}]'
            ),
            FakeResponse(
                content=json.dumps(
                    {
                        "姓名": "张三",
                        "证号": "110101199001011237",
                        "性别": "男",
                        "准驾车型": "C1",
                        "档案编号": "110100123456",
                        "有效期限区间": "2020-01-01至2030-01-01",
                        "类型": "中华人民共和国机动车驾驶证",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    config = replace(
        provider_config(input_mode="vision", max_attempts=1),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        extra_body={"max_tokens": 2048},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(
        load_profiles().get("driver_license"),
        [image_path],
    )

    assert record["姓名"] == "张三"
    assert record["证号"] == "110101199001011237"
    assert record["准驾车型"] == "C1"
    assert record["档案编号"] == "110100123456"
    assert record["有效期限"] == "20200101-20300101"
    assert len(session.calls) == 2
    prompt = session.calls[0]["json"]["messages"][1]["content"][1]["text"]
    assert prompt == document_prompt("driver_license")
    assert "档案编号" in prompt


@pytest.mark.parametrize(
    "title,expected",
    [
        ("电子驾驶证", "电子驾驶证"),
        ("中华人民共和国机动车驾驶证", "中华人民共和国机动车驾驶证"),
    ],
)
def test_hunyuan_driver_title_is_classified_by_backend_rule(
    tmp_path,
    title,
    expected,
) -> None:
    image_path = tmp_path / "driver.jpg"
    image_path.write_bytes(b"driver")
    session = FakeSession(
        [
            FakeResponse(
                content=json.dumps(
                    {
                        "姓名": "张三",
                        "证号": "110101199001011237",
                        "证件标题": title,
                    },
                    ensure_ascii=False,
                )
            )
        ]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        extra_body={},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(
        load_profiles().get("driver_license"),
        [image_path],
    )

    assert record["证件类型"] == expected
    assert "证件标题" not in record


def test_hunyuan_adapter_converts_driver_spotting_json_to_fields(
    tmp_path,
) -> None:
    image_path = tmp_path / "electronic-driver.jpg"
    image_path.write_bytes(b"driver")
    spotting = [
        {"box": [1, 1, 10, 10], "text": "电子驾驶证"},
        {"box": [1, 11, 10, 20], "text": "姓名罗兰"},
        {"box": [1, 21, 10, 30], "text": "准驾车型C1E"},
        {"box": [1, 31, 10, 40], "text": "证号522725199510246140"},
        {"box": [1, 41, 10, 50], "text": "性别女"},
        {"box": [1, 51, 10, 60], "text": "国籍中国"},
        {"box": [1, 61, 10, 70], "text": "档案编号522702242577"},
        {"box": [1, 71, 10, 80], "text": "有效期限2021-04-30至2031-04-30"},
    ]
    session = FakeSession(
        [FakeResponse(content=json.dumps(spotting, ensure_ascii=False))]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        extra_body={},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(
        load_profiles().get("driver_license"),
        [image_path],
    )

    assert record["姓名"] == "罗兰"
    assert record["准驾车型"] == "C1E"
    assert record["证号"] == "522725199510246140"
    assert record["性别"] == "女"
    assert record["国籍"] == "中国"
    assert record["档案编号"] == "522702242577"
    assert record["有效期限"] == "20210430-20310430"
    assert len(session.calls) == 1


def test_hunyuan_adapter_merges_driver_front_and_back_json(tmp_path) -> None:
    front = tmp_path / "front.jpg"
    back = tmp_path / "back.jpg"
    front.write_bytes(b"front")
    back.write_bytes(b"back")
    session = FakeSession(
        [
            FakeResponse(
                content=json.dumps(
                    {
                        "姓名": "李四",
                        "证号": "110101199202022346",
                        "准驾车型": "C1",
                    },
                    ensure_ascii=False,
                )
            ),
            FakeResponse(
                content=json.dumps(
                    {
                        "姓名": "李四",
                        "证号": "性别",
                        "国籍": "档案编号",
                        "副页档案编号": "110100654321",
                    },
                    ensure_ascii=False,
                )
            ),
        ]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        extra_body={},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    records = client.extract_image_records(
        load_profiles().get("driver_license"),
        [front, back],
    )

    assert len(records) == 1
    assert records[0]["姓名"] == "李四"
    assert records[0]["档案编号"] == "110100654321"


def test_hunyuan_adapter_extracts_vehicle_aliases(tmp_path) -> None:
    image = tmp_path / "vehicle.jpg"
    image.write_bytes(b"vehicle")
    session = FakeSession(
        [
            FakeResponse(
                content=json.dumps(
                    {
                        "所有人": "王五",
                        "VIN": "LHGCM82633A123456",
                        "档案编号": "AB1234567890",
                        "号牌号码": "粤A12345",
                    },
                    ensure_ascii=False,
                )
            )
        ]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        extra_body={},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(
        load_profiles().get("vehicle_license"),
        [image],
    )

    assert record["所有人"] == "王五"
    assert record["车辆识别代号"] == "LHGCM82633A123456"
    assert record["档案号码"] == "AB1234567890"
    assert record["号牌号码"] == "粤A12345"


def test_hunyuan_vehicle_syncs_valid_plate_for_each_vin_record(tmp_path) -> None:
    image = tmp_path / "vehicles.jpg"
    image.write_bytes(b"vehicles")
    session = FakeSession(
        [
            FakeResponse(
                content=json.dumps(
                    [
                        {
                            "车辆识别代号": "SALGA2BU8LA598787",
                            "号牌号码": "渝B828YD",
                        },
                        {
                            "车辆识别代号": "SALCA2BU8LA598787",
                            "号牌号码": "SALCA2BU8LA598787",
                        },
                    ],
                    ensure_ascii=False,
                )
            )
        ]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
        extra_body={},
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    records = client.extract_image_records(
        load_profiles().get("vehicle_license"),
        [image],
    )

    assert [record["车辆识别代号"] for record in records] == [
        "SALGA2BU8LA598787",
        "SALCA2BU8LA598787",
    ]
    assert records[0]["车牌号码"] == "渝B828YD"
    assert records[0]["号牌号码"] == "渝B828YD"
    assert records[1]["车牌号码"] == ""


def test_hunyuan_adapter_does_not_assign_unkeyed_back_to_two_holders(
    tmp_path,
) -> None:
    paths = [tmp_path / f"{index}.jpg" for index in range(3)]
    for path in paths:
        path.write_bytes(b"image")
    responses = [
        {
            "姓名": "张三",
            "身份证号": "110101199001011237",
        },
        {
            "姓名": "李四",
            "身份证号": "110101199202022346",
        },
        {"签发机关": "某市公安局", "有效期": "2020.01.01-2040.01.01"},
    ]
    session = FakeSession(
        [
            FakeResponse(content=json.dumps(item, ensure_ascii=False))
            for item in responses
        ]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    records = client.extract_image_records(load_profiles().get("id_card"), paths)

    assert [record["姓名"] for record in records] == ["张三", "李四"]
    assert all(record["签发机关"] == "" for record in records)


def test_hunyuan_adapter_accepts_common_id_card_key_aliases(tmp_path) -> None:
    image = tmp_path / "card.jpg"
    image.write_bytes(b"image")
    response = {
        "公民身份号码": "110101199001011237",
        "出生": "1990年1月1日",
        "有效期限": "2020.01.01-2040.01.01",
    }
    session = FakeSession(
        [FakeResponse(content=json.dumps(response, ensure_ascii=False))]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(load_profiles().get("id_card"), [image])

    assert record["身份证号"] == "110101199001011237"
    assert record["出生日期"] == "19900101"
    assert record["有效期"] == "20200101-20400101"


def test_hunyuan_adapter_skips_one_spotting_response_when_copy_succeeds(
    tmp_path,
) -> None:
    paths = [tmp_path / "bad.jpg", tmp_path / "good.jpg"]
    for path in paths:
        path.write_bytes(b"image")
    session = FakeSession(
        [
            FakeResponse(content='[{"box":[1,2,3,4],"text":"姓名"}]'),
            FakeResponse(content='{"\u59d3\u540d":"\u5f20\u4e09"}'),
        ]
    )
    config = replace(
        provider_config(input_mode="vision"),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(load_profiles().get("id_card"), paths)

    assert record["姓名"] == "张三"


def test_hunyuan_adapter_retries_one_spotting_response(tmp_path) -> None:
    image = tmp_path / "DG13" / "back.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    session = FakeSession(
        [
            FakeResponse(content='[{"box":[1,2,3,4],"text":"有效期"}]'),
            FakeResponse(
                content='{"\u7b7e\u53d1\u673a\u5173":"\u67d0市公安局","\u6709\u6548\u671f":"2020.01.01-2040.01.01"}'
            ),
        ]
    )
    config = replace(
        provider_config(input_mode="vision", max_attempts=2),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(load_profiles().get("id_card"), [image])

    assert record["签发机关"] == "某市公安局"
    assert record["有效期"] == "20200101-20400101"
    assert len(session.calls) == 2


def test_hunyuan_adapter_retries_empty_back_json(tmp_path) -> None:
    image = tmp_path / "DG13" / "back.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    session = FakeSession(
        [
            FakeResponse(content='{"\u8bc1\u4ef6\u7c7b\u578b":"\u5c45\u6c11\u8eab\u4efd\u8bc1"}'),
            FakeResponse(content='{"\u7b7e\u53d1\u673a\u5173":"\u67d0县公安局"}'),
        ]
    )
    config = replace(
        provider_config(input_mode="vision", max_attempts=1),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(load_profiles().get("id_card"), [image])

    assert record["签发机关"] == "某县公安局"
    assert len(session.calls) == 2


def test_hunyuan_adapter_falls_back_to_labeled_text_for_empty_back(
    tmp_path,
) -> None:
    image = tmp_path / "DG13" / "back.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    session = FakeSession(
        [
            FakeResponse(content="[]"),
            FakeResponse(content="[]"),
            FakeResponse(
                content=(
                    "中华人民共和国居民身份证\n"
                    "签发机关 某县公安局\n"
                    "有效期限 2020.01.01-2040.01.01"
                )
            ),
        ]
    )
    config = replace(
        provider_config(input_mode="vision", max_attempts=1),
        response_format=None,
        vision_response_adapter="hunyuan_ocr",
    )
    client = OpenAICompatibleClient(config, session_factory=lambda: session)

    record = client.extract_images(load_profiles().get("id_card"), [image])

    assert record["签发机关"] == "某县公安局"
    assert record["有效期"] == "20200101-20400101"
    assert len(session.calls) == 3


def test_hunyuan_adapter_pairs_multiple_holders_by_side_index() -> None:
    profile = load_profiles().get("id_card")
    page_records = [
        {
            "姓名": "张三",
            "身份证号": "110101199001011237",
            "__hunyuan_side": "front",
            "__hunyuan_side_index": "0",
        },
        {
            "姓名": "李四",
            "身份证号": "110101199202022346",
            "__hunyuan_side": "front",
            "__hunyuan_side_index": "1",
        },
        {
            "签发机关": "甲市公安局",
            "有效期": "20200101-20400101",
            "__hunyuan_side": "back",
            "__hunyuan_side_index": "0",
        },
        {
            "签发机关": "乙市公安局",
            "有效期": "20220202-20420202",
            "__hunyuan_side": "back",
            "__hunyuan_side_index": "1",
        },
    ]

    records = merge_id_card_pages(profile, page_records)

    assert records[0]["姓名"] == "张三"
    assert records[0]["签发机关"] == "甲市公安局"
    assert records[1]["姓名"] == "李四"
    assert records[1]["签发机关"] == "乙市公安局"


def test_hunyuan_driver_merges_masked_and_complete_identity() -> None:
    profile = load_profiles().get("driver_license")
    records = merge_document_pages(
        profile,
        [
            {
                "证号": "141102199110270046",
                "姓名": "贺利梅",
                "档案编号": "411312718817",
                "住址": "",
            },
            {
                "证号": "141102********0046",
                "姓名": "贺利梅",
                "档案编号": "411312718817",
                "住址": "山西省吕梁市离石区城北街道1号",
            },
        ],
    )

    assert len(records) == 1
    assert records[0]["证号"] == "141102199110270046"
    assert records[0]["住址"] == "山西省吕梁市离石区城北街道1号"


def test_hunyuan_driver_does_not_merge_masked_id_without_stable_support() -> None:
    profile = load_profiles().get("driver_license")
    records = merge_document_pages(
        profile,
        [
            {"证号": "141102199110270046", "姓名": "甲"},
            {"证号": "141102********0046", "姓名": "乙"},
        ],
    )

    assert len(records) == 2


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        (
            [{"证件类型": "中华人民共和国机动车行驶证", "所有人": "张三"}],
            "中国行驶证正本",
        ),
        (
            [
                {"证件类型": "中华人民共和国行驶证", "所有人": "张三"},
                {"档案编号": "110101123456"},
            ],
            "中国行驶证正本与副页",
        ),
        (
            [{"整备质量": "1585kg"}],
            "中国行驶证副页",
        ),
        (
            [{"证件类型": "中国行驶证正本与副页"}],
            "中国行驶证正本与副页",
        ),
        (
            [{"所有人": "张三"}],
            "",
        ),
    ],
)
def test_hunyuan_vehicle_type_is_classified_from_page_evidence(
    pages,
    expected,
) -> None:
    profile = load_profiles().get("vehicle_license")
    normalized_pages = [
        record
        for page in pages
        for record in normalize_document_page(profile, page)
    ]

    records = merge_document_pages(profile, normalized_pages)

    assert records[0]["证件类型"] == expected


def test_client_collects_final_content_from_thinking_stream() -> None:
    profile = load_profiles().get("id_card")

    def sse(delta: dict[str, object]) -> str:
        return "data: " + json.dumps(
            {"choices": [{"delta": delta}]},
            ensure_ascii=False,
        )

    stream_lines = [
        sse({"reasoning_content": "分析", "content": None}),
        sse({"content": '{"姓名": "张'}),
        sse({"content": '三"}'}),
        'data: {"choices":[],"usage":{"total_tokens":12}}',
        "data: [DONE]",
    ]
    response = FakeResponse(stream_lines=stream_lines)
    config = replace(
        provider_config(),
        extra_body={
            "enable_thinking": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    session = FakeSession([response])
    client = OpenAICompatibleClient(
        config,
        session_factory=lambda: session,
    )

    result = client.extract(profile, "身份证 OCR")

    assert result["姓名"] == "张三"
    call = session.calls[0]
    assert call["stream"] is True
    assert call["json"]["stream"] is True
    assert call["json"]["enable_thinking"] is True
    assert response.closed is True


def test_client_retries_a_failed_http_attempt(monkeypatch) -> None:
    profile = load_profiles().get("id_card")
    session = FakeSession(
        [
            FakeResponse(status_code=500, text="temporary"),
            FakeResponse(content='{"姓名": "张三"}'),
        ]
    )
    client = OpenAICompatibleClient(
        provider_config(max_attempts=2),
        session_factory=lambda: session,
    )
    monkeypatch.setattr("llm_manager.client.time.sleep", lambda _: None)

    result = client.extract(profile, "OCR")

    assert result["姓名"] == "张三"
    assert len(session.calls) == 2
