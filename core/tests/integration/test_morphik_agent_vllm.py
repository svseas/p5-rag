#!/usr/bin/env python3
"""
Test Morphik Agent with vLLM Gemma 3 12B for tool calling.

This tests whether Gemma 3 12B can:
1. Use retrieve_chunks tool to get Vietnamese contract data
2. Optionally use execute_code tool for comparison
3. Generate Vietnamese answers

PydanticAI is disabled - using standard LiteLLM-based MorphikAgent.
"""

import asyncio
import httpx


async def test_morphik_agent_vietnamese_query():
    """
    Test Morphik Agent with Vietnamese query: 'Thiết bị nào đắt nhất?'

    Expected behavior:
    1. Agent calls retrieve_chunks tool
    2. Agent analyzes data (might use execute_code)
    3. Agent generates Vietnamese answer
    """

    print("\n" + "=" * 80)
    print("MORPHIK AGENT TEST - vLLM Gemma 3 12B Tool Calling")
    print("=" * 80)
    print("\nQuestion: Thiết bị nào đắt nhất?")
    print("Expected: Bộ khuếch đại at 2,346,682,000 VND\n")

    print("🔄 Sending query to /agent endpoint...")
    print("   Using: vLLM Gemma 3 12B (LiteLLM + MorphikAgent)")
    print("   PydanticAI: DISABLED\n")

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                'http://localhost:8000/agent',
                json={
                    "query": "Thiết bị nào đắt nhất?",
                    "end_user_id": "test-user"
                }
            )
            response.raise_for_status()
            data = response.json()

            print("=" * 80)
            print("RESPONSE RECEIVED")
            print("=" * 80)

            # Extract tool history
            tool_history = data.get('tool_history', [])
            print(f"\n📋 Tools Called: {len(tool_history)}")

            if tool_history:
                print("\nTool Call History:")
                for i, tool_call in enumerate(tool_history, 1):
                    tool_name = tool_call.get('tool_name', 'unknown')
                    print(f"   {i}. {tool_name}")

                    # Check if retrieve_chunks was called
                    if tool_name == 'retrieve_chunks':
                        print("      ✅ Successfully called retrieve_chunks")

                    # Check if execute_code was called
                    if tool_name == 'execute_code':
                        print("      ✅ Used execute_code for computation")
                        code = tool_call.get('input', {}).get('code', '')
                        if code:
                            print(f"      Code snippet (first 200 chars):")
                            print(f"      {code[:200]}")
            else:
                print("   ⚠️  NO TOOLS CALLED")
                print("   This means Gemma 3 12B did not use function calling")
                print("   The model may have answered directly without retrieval")

            # Extract sources (retrieved chunks)
            sources = data.get('sources', [])
            print(f"\n📄 Sources Retrieved: {len(sources)}")

            if sources:
                print("\nSample sources:")
                for i, source in enumerate(sources[:2], 1):
                    doc_name = source.get('documentName', 'unknown')
                    score = source.get('score', 0.0)
                    content_preview = source.get('content', '')[:100]
                    print(f"   {i}. {doc_name} (score: {score:.3f})")
                    print(f"      {content_preview}...")

            # Extract answer
            answer = data.get('response', '')
            print(f"\n💬 Generated Answer:")
            print(f"   {answer}\n")

            # Validate answer
            answer_lower = answer.lower()

            validations = []

            # Check if mentions "khuếch đại"
            if "khuếch đại" in answer_lower:
                validations.append("✅ Mentions 'Bộ khuếch đại'")
            else:
                validations.append("❌ Does NOT mention 'Bộ khuếch đại'")

            # Check if mentions price
            has_price = any([
                "2.346.682.000" in answer,
                "2,346,682,000" in answer,
                "2346682000" in answer,
                "2.3 tỷ" in answer_lower,
                "hai tỷ ba trăm" in answer_lower
            ])

            if has_price:
                validations.append("✅ Mentions correct price")
            else:
                validations.append("⚠️  Price not clearly mentioned")

            print("=" * 80)
            print("VALIDATION RESULTS")
            print("=" * 80)
            for validation in validations:
                print(f"   {validation}")

            print("\n" + "=" * 80)
            print("TOOL CALLING ASSESSMENT")
            print("=" * 80)

            if len(tool_history) == 0:
                print("❌ FAIL - Gemma 3 12B did NOT use tool calling")
                print("\nThis means:")
                print("  • Model answered directly without using retrieve_chunks")
                print("  • vLLM function calling may not work with Gemma 3 12B")
                print("  • Consider using Qwen3:32b with Ollama instead")

            elif 'retrieve_chunks' in [t.get('tool_name') for t in tool_history]:
                print("✅ SUCCESS - Gemma 3 12B used tool calling correctly!")
                print("\nThe model:")
                print(f"  • Called {len(tool_history)} tool(s)")
                print("  • Retrieved data using retrieve_chunks")

                if 'execute_code' in [t.get('tool_name') for t in tool_history]:
                    print("  • Used execute_code for computation (BONUS!)")

                if "khuếch đại" in answer_lower and has_price:
                    print("  • Generated correct Vietnamese answer ✅")
                else:
                    print("  • Answer may be incomplete ⚠️")
            else:
                print("⚠️  PARTIAL - Called tools but not retrieve_chunks")
                print(f"   Tools called: {[t.get('tool_name') for t in tool_history]}")

            print("=" * 80)

            return len(tool_history) > 0

        except httpx.HTTPStatusError as e:
            print(f"\n❌ HTTP ERROR: {e}")
            print(f"   Status: {e.response.status_code}")
            print(f"   Response: {e.response.text[:500]}")
            return False

        except httpx.TimeoutException:
            print("\n❌ TIMEOUT: Agent took too long (>5 minutes)")
            print("   Possible issues:")
            print("   • vLLM is slow with Gemma 3 12B")
            print("   • Model is stuck in a loop")
            print("   • Tool execution is failing")
            return False

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    print("\n🚀 TESTING MORPHIK AGENT WITH vLLM")
    print("Configuration:")
    print("  • Model: Gemma 3 12B IT")
    print("  • Backend: vLLM (OpenAI-compatible API)")
    print("  • Agent: MorphikAgent (LiteLLM)")
    print("  • PydanticAI: DISABLED")
    print()

    # Wait for morphik to be ready
    print("⏳ Waiting for morphik container to be ready...")
    await asyncio.sleep(10)

    success = await test_morphik_agent_vietnamese_query()

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if success:
        print("✅ GEMMA 3 12B CAN USE TOOL CALLING WITH vLLM")
        print("\nRecommendation:")
        print("  • Keep vLLM Gemma 3 12B as agent model")
        print("  • Morphik Agent with tool calling works!")
        print("  • PydanticAI extraction not needed")
    else:
        print("❌ GEMMA 3 12B CANNOT USE TOOL CALLING RELIABLY")
        print("\nRecommendation:")
        print("  • Switch back to Ollama Qwen3:32b for agent")
        print("  • Use vLLM Gemma 3 only for Vietnamese generation")
        print("  • Tool calling requires better model support")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
