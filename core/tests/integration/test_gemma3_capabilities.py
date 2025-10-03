"""Test Gemma 3 12B IT capabilities in isolation:
1. Structured output / function calling
2. Vietnamese language understanding
3. PydanticAI integration
"""

import asyncio
import json
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import httpx


# Test 1: Simple structured output
class EquipmentItem(BaseModel):
    """Equipment item from Vietnamese contract."""
    name: str = Field(description="Equipment name in Vietnamese")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    quantity: Optional[float] = Field(None, description="Quantity")
    unit_price: Optional[float] = Field(None, description="Unit price in VND")
    total_price: Optional[float] = Field(None, description="Total price in VND")


class ExtractedData(BaseModel):
    """Structured data extracted from Vietnamese contract."""
    equipment_items: list[EquipmentItem] = Field(
        default_factory=list,
        description="Equipment items from the contract"
    )
    contract_number: Optional[str] = Field(None, description="Contract number")
    total_value: Optional[float] = Field(None, description="Total contract value in VND")


async def test_structured_output():
    """Test 1: Structured output with Vietnamese contract data."""
    print("\n" + "="*80)
    print("TEST 1: Structured Output - Vietnamese Contract Extraction")
    print("="*80)

    # Vietnamese contract excerpt with equipment table
    vietnamese_contract = """
    HỢP ĐỒNG MUA BÁN
    Số: 16/2022/HĐMB

    DANH MỤC HÀNG HÓA:

    | TT | DANH MỤC HÀNG HÓA | ĐVT   | SL | ĐƠN GIÁ (VNĐ) | THÀNH TIỀN (VNĐ) |
    |----|-------------------|-------|----|--------------:|------------------:|
    | 1  | Đèn điện từ       | Cái   | 1  | 353.066.000   | 353.066.000       |
    | 2  | Đèn điều chế      | Cái   | 2  | 65.000.000    | 130.000.000       |
    | 3  | Bộ chia           | Bộ    | 8  | 120.000.000   | 960.000.000       |
    | 4  | Bộ khuếch đại     | Bộ    | 1  | 2.346.682.000 | 2.346.682.000     |

    Tổng giá trị: 11.878.135.831 đồng
    """

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
        output_type=ExtractedData,
        system_prompt="""Extract equipment items from Vietnamese contract tables.

        Extract ALL items with their prices and specifications.
        Preserve exact numbers - DO NOT modify."""
    )

    try:
        result = await agent.run(
            f"Extract all equipment from this contract:\n\n{vietnamese_contract}"
        )

        print("\n✅ Structured Output SUCCESS")
        print(f"Extracted {len(result.output.equipment_items)} equipment items:")
        for item in result.output.equipment_items:
            print(f"  - {item.name}: {item.unit_price:,.0f} VND (quantity: {item.quantity})")
        print(f"Contract number: {result.output.contract_number}")
        print(f"Total value: {result.output.total_value:,.0f} VND" if result.output.total_value else "Total value: Not extracted")

        return True

    except Exception as e:
        print(f"\n❌ Structured Output FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await http_client.aclose()


async def test_vietnamese_generation():
    """Test 2: Vietnamese language generation."""
    print("\n" + "="*80)
    print("TEST 2: Vietnamese Language Generation")
    print("="*80)

    http_client = httpx.AsyncClient(timeout=120.0)

    try:
        response = await http_client.post(
            'http://localhost:8080/v1/chat/completions',
            json={
                "model": "/models/gemma-3-12b-it",
                "messages": [
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI chuyên về hợp đồng tiếng Việt."
                    },
                    {
                        "role": "user",
                        "content": "Thiết bị nào đắt nhất trong danh sách sau?\n\n1. Đèn điện từ: 353.066.000 VND\n2. Bộ chia: 960.000.000 VND\n3. Bộ khuếch đại: 2.346.682.000 VND\n\nTrả lời bằng tiếng Việt."
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 200
            }
        )
        response.raise_for_status()

        result = response.json()
        vietnamese_answer = result['choices'][0]['message']['content']

        print("\n✅ Vietnamese Generation SUCCESS")
        print(f"Question: Thiết bị nào đắt nhất?")
        print(f"Answer: {vietnamese_answer}")

        # Check if answer mentions "Bộ khuếch đại" (the most expensive)
        if "khuếch đại" in vietnamese_answer.lower():
            print("✅ Correctly identified most expensive equipment")
            return True
        else:
            print("⚠️  Answer may not be correct")
            return False

    except Exception as e:
        print(f"\n❌ Vietnamese Generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await http_client.aclose()


async def test_function_calling():
    """Test 3: Function calling capability."""
    print("\n" + "="*80)
    print("TEST 3: Function Calling / Tool Use")
    print("="*80)

    http_client = httpx.AsyncClient(timeout=120.0)

    tools = [{
        "type": "function",
        "function": {
            "name": "find_most_expensive",
            "description": "Find the most expensive equipment from a Vietnamese contract",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_name": {
                        "type": "string",
                        "description": "Name of the most expensive equipment in Vietnamese"
                    },
                    "price": {
                        "type": "number",
                        "description": "Price in VND"
                    }
                },
                "required": ["equipment_name", "price"]
            }
        }
    }]

    try:
        response = await http_client.post(
            'http://localhost:8080/v1/chat/completions',
            json={
                "model": "/models/gemma-3-12b-it",
                "messages": [
                    {
                        "role": "user",
                        "content": "Find the most expensive equipment:\n1. Đèn điện từ: 353,066,000 VND\n2. Bộ khuếch đại: 2,346,682,000 VND\n3. Bộ chia: 960,000,000 VND"
                    }
                ],
                "tools": tools,
                "tool_choice": "auto"
            }
        )
        response.raise_for_status()

        result = response.json()
        message = result['choices'][0]['message']

        if 'tool_calls' in message and message['tool_calls']:
            tool_call = message['tool_calls'][0]
            args = json.loads(tool_call['function']['arguments'])

            print("\n✅ Function Calling SUCCESS")
            print(f"Tool called: {tool_call['function']['name']}")
            print(f"Arguments: {json.dumps(args, ensure_ascii=False, indent=2)}")

            return True
        else:
            print("\n⚠️  No tool calls made")
            print(f"Response: {message.get('content', 'No content')}")
            return False

    except Exception as e:
        print(f"\n❌ Function Calling FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await http_client.aclose()


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GEMMA 3 12B IT - CAPABILITY TESTS")
    print("="*80)
    print("\nMake sure vLLM is running with Gemma 3 12B:")
    print("  docker compose up -d vllm")
    print("\nWaiting 5 seconds for vLLM to be ready...")
    await asyncio.sleep(5)

    results = []

    # Test 1: Structured Output
    results.append(("Structured Output", await test_structured_output()))

    # Test 2: Vietnamese Generation
    results.append(("Vietnamese Generation", await test_vietnamese_generation()))

    # Test 3: Function Calling
    results.append(("Function Calling", await test_function_calling()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(result[1] for result in results)
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Gemma 3 12B is ready for production!")
    else:
        print("⚠️  SOME TESTS FAILED - Check configuration")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
