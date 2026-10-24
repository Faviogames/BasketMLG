import asyncio
import os
import sys
from playwright.async_api import async_playwright

async def test_access():
    event_id = "16685165"
    print(f"Testing access to SofaScore event {event_id} with Chrome persistent context...")
    
    user_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.browser_profile'))
    os.makedirs(user_data_dir, exist_ok=True)
    
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            channel="chrome",
            headless=False,
            args=[
                '--no-first-run',
                '--no-default-browser-check',
                '--disable-blink-features=AutomationControlled'
            ],
            viewport={'width': 1280, 'height': 720}
        )
        page = context.pages[0] if context.pages else await context.new_page()
            
        print("1. Establishing session at sofascore.com/basketball...")
        await page.goto("https://www.sofascore.com/basketball", wait_until="domcontentloaded", timeout=30000)
        print(f"Current page URL: {page.url}")
        
        if "captcha" in page.url:
            print("[INFO] Captcha detectado. Esperando a que Cloudflare Turnstile verifique...")
            for i in range(20):
                await asyncio.sleep(1)
                if "captcha" not in page.url:
                    print(f"✅ Captcha verificado y redirigido a {page.url} tras {i+1}s!")
                    break
            else:
                print(f"[WARN] Sigue en captcha tras 20s. URL actual: {page.url}")
        else:
            print("Sesión establecida directamente sin captcha.")
            
        await asyncio.sleep(2)
        
        print("Testing live endpoints inside page:")
        for u in ['/api/v1/sport/basketball/live-tournaments', '/api/v1/sport/basketball/events/live']:
            res = await page.evaluate(f'''async () => {{
                try {{
                    const r = await fetch("{u}");
                    return {{ status: r.status, ok: r.ok, text: (await r.text()).slice(0, 100) }};
                }} catch (e) {{
                    return {{ error: e.message }};
                }}
            }}''')
            print(f"  {u} -> {res}")
            
        print(f"\n2. Fetching event data for {event_id}...")
        event_url = f"https://www.sofascore.com/api/v1/event/{event_id}"
        event_res = await page.evaluate(f'''async () => {{
            try {{
                const r = await fetch("{event_url}");
                return await r.json();
            }} catch (e) {{
                return {{ error: e.message }};
            }}
        }}''')
        event = event_res.get('event', {})
        print(f"Match: {event.get('homeTeam', {}).get('name')} vs {event.get('awayTeam', {}).get('name')}")
        print(f"Tournament: {event.get('tournament', {}).get('name')}")
        print(f"Status: {event.get('status', {}).get('description')}")
        print(f"Home score: {event.get('homeScore', {})}")
        print(f"Away score: {event.get('awayScore', {})}")
        print(f"Time: {event.get('time', {})}")

        print(f"\n3. Fetching stats for event {event_id}...")
        stats_url = f"https://www.sofascore.com/api/v1/event/{event_id}/statistics"
        
        # Try both evaluate fetch and goto
        stats_res = await page.evaluate(f'''async () => {{
            try {{
                const r = await fetch("{stats_url}");
                return {{ status: r.status, ok: r.ok, text: await r.text() }};
            }} catch (e) {{
                return {{ status: 0, ok: false, error: e.message }};
            }}
        }}''')
        
        print(f"Fetch inside page: status={stats_res.get('status')} ok={stats_res.get('ok')}")
        if stats_res.get('ok'):
            print("SUCCESS! Stats received via page fetch!")
            import json
            try:
                data = json.loads(stats_res['text'])
                periods = data.get('statistics', [])
                print(f"Statistics periods available: {len(periods)}")
                for p in periods:
                    print(f"  Period: {p.get('period')}, Groups: {len(p.get('groups', []))}")
            except Exception as e:
                print(f"Error parsing json: {e}")
        else:
            print(f"Fetch not ok, trying page.goto({stats_url})...")
            goto_res = await page.goto(stats_url, wait_until="domcontentloaded")
            print(f"Goto status: {goto_res.status}")
            body = await page.evaluate("() => document.body.innerText")
            print(f"Body preview: {body[:300]}")
            
        await context.close()
        print("Test finished.")

if __name__ == "__main__":
    asyncio.run(test_access())
