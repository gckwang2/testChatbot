import asyncio
from putergenai import PuterClient

async def check_auth():
    async with PuterClient() as client:
        try:
            # TRY THIS: Use only the username, no email format
            await client.login("YOUR_EXACT_USERNAME", "YOUR_PASSWORD")
            print("✅ Handshake successful!")
        except Exception as e:
            print(f"❌ Handshake failed: {e}")
            print("TIP: Ensure you aren't using an email address as the username.")

# In Streamlit, use this to avoid the "loop already running" error:
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(check_auth())
