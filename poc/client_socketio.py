import asyncio
import json
from socketio.async_client import AsyncClient


async def chat_client():
    sio = AsyncClient()
    try:
        await sio.connect("http://localhost:8000", socketio_path="socket.io")
        print("Connected to server")
        await sio.emit("simple_message", "Hello Server!")

        @sio.on("chat_response")
        async def handle_chat_response(data):
            try:
                if "response" in data:
                    print("Server:", data["response"])
                elif "error" in data:
                    print("Error:", data["error"])
                else:
                    print("Unexpected response:", data)
            except Exception as e:
                print(f"Error handling response: {e}")

        @sio.on("simple_response")
        async def handle_simple_response(data):
            print(f"Message from server: {data}")

        while True:
            user_input = input("Enter message (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                break

            await sio.emit("chat_message", {"message": user_input})

        await sio.disconnect()
        print("Disconnected from server.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if sio.connected:
            await sio.disconnect()


if __name__ == "__main__":
    asyncio.run(chat_client())
