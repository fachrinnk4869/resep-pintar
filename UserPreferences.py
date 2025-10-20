import os
import asyncio
import aiohttp
from together import Together


class UserPreferences:
    def __init__(self):
        self.client = Together()

    async def get_possible_ingredients(self, ingredients_text: str, user_preference: str):
        """
        Async version — menggunakan Together API untuk memprediksi bahan tambahan,
        tetapi dijalankan secara non-blocking.
        """
        prompt = f"""
        Kamu adalah asisten kuliner yang membantu membuat resep masakan.

        Berikut data dari pengguna:
        - Daftar bahan yang sudah dimiliki: {ingredients_text if ingredients_text.strip() else "(belum ada bahan)"}
        - Preferensi rasa pengguna: {user_preference}

        Berdasarkan informasi di atas, berikan 3–5 bahan tambahan yang cocok dengan preferensi pengguna.
        Berikan hanya daftar bahan dalam format dipisahkan koma, tanpa penjelasan tambahan atau angka.
        """

        try:
            loop = asyncio.get_event_loop()

            # Jalankan pemanggilan API di executor (karena Together client belum async)
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[
                        {"role": "system",
                            "content": "Kamu adalah asisten kuliner yang ramah dan ahli masak."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6
                )
            )

            text = response.choices[0].message.content.strip()
            ingredients = [i.strip() for i in text.split(",") if i.strip()]
            return ingredients

        except Exception as e:
            print("Error generating preferences:", e)
            return []
