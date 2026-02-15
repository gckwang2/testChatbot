import asyncio
from putergenai import PuterClient

async def test_puter():
    async with PuterClient() as client:
        # Step 1: Login
        await client.login("YOUR_USERNAME", "YOUR_PASSWORD")
        print("✅ Login Successful")
        
        # Step 2: Simple Chat (No Vector Search)
        print("⏳ Sending test chat...")
        try:
            # Try a smaller model first to verify access
            resp = await client.ai_chat("Hello", options={"model": "gpt-4o-mini"})
            print(f"✅ Response received: {resp}")
        except Exception as e:
            print(f"❌ Chat failed: {e}")

asyncio.run(test_puter())
