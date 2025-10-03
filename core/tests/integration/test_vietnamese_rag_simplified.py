#!/usr/bin/env python3
"""
Integration test for simplified Vietnamese RAG (2-stage: Retrieve + Generate).

Tests that Gemma 3 12B can answer Vietnamese questions directly from retrieved chunks
without needing structured extraction.

This validates the hypothesis: We don't need Stage 2 (extraction) - just retrieve
chunks and let Gemma 3 answer the question.
"""

import asyncio
import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# Real chunk from hop_dong_mua_ban_2022.pdf containing equipment table
REAL_CHUNK_WITH_EQUIPMENT = """
### **Phục lục I DANH MỤC HÀNG HÓA**

| (Kèm theo Hợp đồng số /2022/HĐMB ngày 06/01/2022) |

| TT | DANH MỤC HÀNG HÓA | QUY CÁCH (mã số, ký hiệu) | XUẤT XỨ | ĐVT   | SL | ĐƠN GIÁ (VNĐ) | THÀNH TIỀN (VNĐ) |
|----|-------------------|---------------------------|---------|-------|----|--------------:|------------------:|
| 1  | Đèn điện từ       | ГМИ-46Б                   | Nga     | Cái   | 1  | 353.066.000   | 353.066.000       |
| 2  | Đèn điều chế      | ГТК1 1000/25              | Nga     | Cái   | 2  | 65.000.000    | 130.000.000       |
| 3  | Bộ chia           | ИЗАА47                    | Nga     | Bộ    | 8  | 120.000.000   | 960.000.000       |
| 4  | Bộ khuếch đại     | УМ-100                    | Nga     | Bộ    | 1  | 2.346.682.000 | 2.346.682.000     |
| 5  | Cảm biến          | AVM58-N                   | Nga     | Cái   | 1  | 28.851.000    | 28.851.000        |
| 6  | Chấn từ           | ИЗАА09                    | Nga     | Cái   | 7  | 75.000.000    | 525.000.000       |
| 7  | Chấn tử           | ИЗАА22                    | Nga     | Cái   | 11 | 60.000.000    | 660.000.000       |
| 8  | Chấn tử           | ИЗАА22М                   | Nga     | Cái   | 4  | 56.000.000    | 224.000.000       |

*Bằng chữ: Mười một tỷ, tám trăm bảy mươi tám triệu, một trăm ba mươi lăm nghìn tám trăm ba mươi mốt đồng.*
"""


@pytest.mark.asyncio
async def test_gemma3_direct_answer_from_chunks():
    """
    Test that Gemma 3 12B can answer 'Thiết bị nào đắt nhất?' directly from chunks.

    This validates we can skip extraction and go straight to generation.
    """

    print("\n" + "=" * 80)
    print("TEST: Simplified 2-Stage RAG (Retrieve → Generate)")
    print("=" * 80)
    print("\nQuestion: Thiết bị nào đắt nhất?")
    print("Expected Answer: Bộ khuếch đại at 2,346,682,000 VND\n")

    # Setup Gemma 3 12B model
    http_client = httpx.AsyncClient(timeout=120.0)

    model = OpenAIChatModel(
        model_name='/models/gemma-3-12b-it',
        provider=OpenAIProvider(
            base_url='http://localhost:8080/v1',
            api_key='dummy',
            http_client=http_client
        )
    )

    # Create a simple agent for Vietnamese Q&A
    agent = Agent(
        model,
        system_prompt="""Bạn là trợ lý AI chuyên về hợp đồng tiếng Việt.

Trả lời câu hỏi dựa trên thông tin được cung cấp trong tài liệu.
Chỉ sử dụng thông tin có trong tài liệu, không bịa đặt.
Trả lời bằng tiếng Việt, ngắn gọn và chính xác."""
    )

    # Simulate retrieved chunks (Stage 1 result)
    user_prompt = f"""Dựa trên thông tin trong tài liệu sau:

{REAL_CHUNK_WITH_EQUIPMENT}

Câu hỏi: Thiết bị nào đắt nhất?"""

    print("🔄 Generating answer with Gemma 3 12B...")

    try:
        result = await agent.run(user_prompt)
        answer = result.output

        print(f"\n✅ Generated Answer:")
        print(f"   {answer}\n")

        # Validate answer
        answer_lower = answer.lower()

        assert "khuếch đại" in answer_lower, "Answer should mention 'khuếch đại'"

        # Check if price is mentioned (various formats)
        has_price = any([
            "2.346.682.000" in answer,
            "2,346,682,000" in answer,
            "2346682000" in answer,
            "2.3 tỷ" in answer_lower,
            "hai tỷ ba trăm" in answer_lower
        ])

        assert has_price, "Answer should mention the price"

        print("✅ VALIDATION PASSED:")
        print("   - Correctly identified 'Bộ khuếch đại'")
        print("   - Mentioned the price")
        print("\n" + "=" * 80)
        print("🎯 CONCLUSION: Gemma 3 12B can answer directly from chunks!")
        print("   → Stage 2 (extraction) is NOT needed for Q&A")
        print("=" * 80)

        return True

    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        print("   Answer did not contain expected information")
        return False

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_gemma3_list_all_equipment():
    """Test that Gemma 3 can list all equipment when asked."""

    print("\n" + "=" * 80)
    print("TEST: List all equipment (no extraction needed)")
    print("=" * 80)

    http_client = httpx.AsyncClient(timeout=120.0)

    model = OpenAIChatModel(
        model_name='/models/gemma-3-12b-it',
        provider=OpenAIProvider(
            base_url='http://localhost:8080/v1',
            api_key='dummy',
            http_client=http_client
        )
    )

    agent = Agent(
        model,
        system_prompt="""Bạn là trợ lý AI chuyên về hợp đồng tiếng Việt.
Trả lời bằng tiếng Việt, liệt kê đầy đủ theo yêu cầu."""
    )

    user_prompt = f"""Dựa trên thông tin trong tài liệu sau:

{REAL_CHUNK_WITH_EQUIPMENT}

Câu hỏi: Liệt kê tất cả thiết bị và giá của chúng."""

    print("🔄 Asking Gemma 3 to list all equipment...")

    try:
        result = await agent.run(user_prompt)
        answer = result.output

        print(f"\n✅ Generated Answer:\n")
        print(answer)
        print()

        # Check if key equipment is mentioned
        answer_lower = answer.lower()
        mentioned_equipment = 0

        equipment_to_check = [
            "đèn điện từ",
            "bộ chia",
            "khuếch đại",
            "cảm biến"
        ]

        for equipment in equipment_to_check:
            if equipment in answer_lower:
                mentioned_equipment += 1
                print(f"   ✓ Found: {equipment}")

        print(f"\n   Total: {mentioned_equipment}/{len(equipment_to_check)} equipment items found")

        assert mentioned_equipment >= 3, "Should mention at least 3 equipment items"

        print("\n✅ VALIDATION PASSED: Can list equipment without extraction")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False

    finally:
        await http_client.aclose()


if __name__ == "__main__":
    """Run tests standalone."""

    print("\n🚀 SIMPLIFIED VIETNAMESE RAG TESTS")
    print("Testing hypothesis: We can skip extraction and answer directly\n")

    async def run_all_tests():
        test1 = await test_gemma3_direct_answer_from_chunks()
        test2 = await test_gemma3_list_all_equipment()

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"1. Direct Answer Test:  {'✅ PASS' if test1 else '❌ FAIL'}")
        print(f"2. List Equipment Test: {'✅ PASS' if test2 else '❌ FAIL'}")

        if test1 and test2:
            print("\n🎉 ALL TESTS PASSED")
            print("\n📋 RECOMMENDATION:")
            print("   Remove Stage 2 (extraction) from vietnamese_agent.py")
            print("   Use simplified 2-stage: Retrieve → Generate")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("   Extraction may still be needed")

        print("=" * 80)

    asyncio.run(run_all_tests())
