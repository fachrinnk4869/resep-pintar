import os
from together import Together

class UserPreferences:
    def __init__(self):
        self.client = Together()

    def get_possible_ingredients(self, ingredients_text: str, user_preference: str):
        """
        Combine user's ingredient list (if any) with user preference
        and ask the LLM to infer what possible additional ingredients
        would fit that preference.
        """
        prompt = f"""
        Kamu adalah asisten kuliner yang membantu membuat resep masakan.

        Berikut data dari pengguna:
        - Daftar bahan yang sudah dimiliki: {ingredients_text if ingredients_text.strip() else "(belum ada bahan)"}
        - Preferensi rasa pengguna: {user_preference}

        Berdasarkan informasi di atas, berikan 3–5 bahan tambahan yang cocok dengan preferensi pengguna.
        Contoh:
        - Jika pengguna suka makanan pedas → tambahkan bahan seperti cabai, saus sambal, lada, dll.
        - Jika pengguna suka makanan manis → tambahkan bahan seperti gula, madu, cokelat, susu kental manis, dll.
        - Jika pengguna suka gurih → tambahkan bahan seperti keju, mentega, kaldu ayam, saus tiram, dll.

        Berikan hanya daftar bahan dalam format dipisahkan koma,
        tanpa penjelasan tambahan atau angka.
        """

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {"role": "system", "content": "Kamu adalah asisten kuliner yang ramah dan ahli masak."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )

            text = response.choices[0].message.content.strip()
            # Split comma-separated result into list
            ingredients = [i.strip() for i in text.split(",") if i.strip()]
            return ingredients

        except Exception as e:
            print("Error generating preferences:", e)
            return []
