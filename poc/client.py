import asyncio
import json
import websockets


async def chat_client():
    uri = "ws://localhost:8000/chat"  # Replace with your server address

    async with websockets.connect(uri, ping_interval=1000) as websocket:
        print("Connected to server")

        while True:
            user_input = input("Enter message: ")
            if user_input.lower() == "exit":
                break

            await websocket.send(json.dumps({"message": user_input}))

            response = await websocket.recv()
            try:
                data = json.loads(response)
                if "response" in data:
                    print("Server:", data["response"])
                elif "error" in data:
                    print("Error:", data["error"])
                else:
                    print("Unexpected response:", response)
            except json.JSONDecodeError:
                print("Error decoding JSON response:", response)


if __name__ == "__main__":
    asyncio.run(chat_client())
