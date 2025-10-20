from UserPreferences import UserPreferences
import json
import time
import gradio as gr
from AlgorithmClass import AlgorithmClass
import asyncio
from helper import MultimodalModel
from rag import Datahandle
data_handler = Datahandle()
preferences_handler = UserPreferences()

# ini ganti dengan rag beneran dan embedding custom # <-- disini bay


# =============================================================

# --- Bagian 1: Definisikan JavaScript secara terpisah ---
# JavaScript ini akan mengamati perubahan pada DOM.
# Ketika carousel kita ditambahkan oleh Gradio, script ini akan berjalan
# dan membuat tombolnya fungsional.
javascript_code = """
() => {
    // Fungsi untuk menginisialisasi carousel
    function setupCarousel(carouselContainer) {
        if (carouselContainer.querySelector('.prev-btn')) {
            let slideIndex = 0;
            const slides = carouselContainer.getElementsByClassName("carousel-slide");
            const prevBtn = carouselContainer.querySelector(".prev-btn");
            const nextBtn = carouselContainer.querySelector(".next-btn");

            const showSlide = (n) => {
                for (let i = 0; i < slides.length; i++) {
                    slides[i].style.display = "none";
                }
                slides[n].style.display = "block";
            };

            prevBtn.onclick = () => {
                slideIndex = (slideIndex - 1 + slides.length) % slides.length;
                showSlide(slideIndex);
            };

            nextBtn.onclick = () => {
                slideIndex = (slideIndex + 1) % slides.length;
                showSlide(slideIndex);
            };

            showSlide(slideIndex);
        }
    }

    // Gunakan MutationObserver untuk mendeteksi kapan Gradio menambahkan HTML carousel
    const observer = new MutationObserver((mutationsList, observer) => {
        for(const mutation of mutationsList) {
            if (mutation.type === 'childList') {
                const carousel = document.querySelector('.carousel-container');
                if (carousel) {
                    setupCarousel(carousel);
                    // observer.disconnect(); // Opsional: hentikan pengamatan jika hanya perlu sekali
                }
            }
        }
    });

    // Mulai mengamati perubahan pada body dokumen
    observer.observe(document.body, { childList: true, subtree: true });

    // Coba jalankan sekali saat awal, siapa tahu elemennya sudah ada
    const initialCarousel = document.querySelector('.carousel-container');
    if (initialCarousel) {
        setupCarousel(initialCarousel);
    }
}
"""
# --- Bagian 2: Fungsi Logika untuk Menghasilkan HTML ---
# Fungsi ini SEKARANG menerima input dari UI Gradio.
# Perhatikan tidak ada lagi tag <script> di sini.


def render_steps(input_text=None, algorithm_state=None):
    if not input_text or input_text.strip() == "":
        return "<p>Please enter some text and press 'Generate' to see the recipe.</p>"

    # Simulasi hasil dari algoritma Anda
    result = algorithm_state.get_recipe()
    # print("Rendering result steps:", result['steps'])
    # print("Rendering result steps:", type(result['steps']))

    # Ambil data
    cover_img = result.get("image")
    title = result.get("title")
    ingredients = result.get("ingredients", [])

    # Slide 1: Cover
    slide1 = f"""
    <div class="carousel-slide main-scroll">
        <img src="{cover_img}" class="cover-img">
        <h2>{title}</h2>
    </div>
    """

    # Slide 2: Ingredients
    ing_html = "".join([f"<li>{ing}</li>" for ing in ingredients])
    slide2 = f"""
    <div class="carousel-slide ingredients-scroll">
        <h3>Ingredients</h3>
        <ul>{ing_html}</ul>
    </div>
    """

    # Slide 3: Steps
    steps_html_content = ""
    for idx, step in enumerate(result['steps'], start=1):
        # --- Bagian Baru ---
        # 1. Buat blok HTML khusus untuk gambar-gambar
        images_html = ""
        # Pastikan ada gambar sebelum melakukan loop
        # print(step)
        if step['images']:
            # Loop melalui SETIAP URL gambar di dalam list step['images']
            for img_url in step['images']:
                images_html += f'<img src="{img_url}" class="step-image" alt="Image for step {idx}">'

        # 2. Gabungkan blok gambar dengan teks langkah dalam satu item
        steps_html_content += f"""
        <div class="step-item">
            <div class="step-images-container">
                {images_html}
            </div>
            <div class="step-text">
                <b>Step {idx}:</b> {step['text']}
            </div>
        </div>
        """
    slide3 = f"""
    <div class="carousel-slide">
        <h3>Steps</h3>
        <div class="steps-scroll">{steps_html_content}</div>
    </div>
    """

    # Gabungkan jadi carousel (tanpa <style> dan <script>)
    html = f"""
    <div class="carousel-container">
        {slide1}
        {slide2}
        {slide3}
        <button class="prev-btn">&#10094;</button>
        <button class="next-btn">&#10095;</button>
    </div>
    """
    return html


def upload_file(file):
    if file is None:
        return None
    return file.name  # kasih path biar bisa dipreview di gr.Image


def text_replace(file):
    model = MultimodalModel()
    return model.generate(file)


# --- Functions that receive per-user algorithm via State ---
async def generate_recipe(input_text, user_preferences, algorithm_state):
    loop = asyncio.get_event_loop()
    algorithm_state.reset()

    new_input = input_text
    if user_preferences:
        possible_ingre = await preferences_handler.get_possible_ingredients(input_text, user_preferences)
        if possible_ingre:
            new_input += f"\nBahan tambahan yang mungkin cocok: {', '.join(possible_ingre)}"

    # Panggilan berat (misal embedding atau retriever)
    dense_input = await data_handler.get_dense_input(new_input)
    await loop.run_in_executor(None, algorithm_state.mapping_input, new_input, dense_input)

    recipes = await data_handler.get_recipes(input_text)
    dense_all, dense_ingr = await loop.run_in_executor(None, data_handler.get_embeddings_recipe, recipes)

    await loop.run_in_executor(None, algorithm_state.mapping_output, recipes, dense_all, dense_ingr)
    await loop.run_in_executor(None, algorithm_state.first_generate_recipe)

    html = await loop.run_in_executor(None, render_steps, input_text, algorithm_state)
    return (
        html,
        gr.update(visible=True),
        gr.update(visible=True),
        algorithm_state,
    )


async def next_recommendation(input_text, rating, algorithm_state):
    loop = asyncio.get_event_loop()

    start_time = time.time()
    dense_input = await data_handler.get_dense_input(input_text)
    end_time = time.time()
    print(f"Time to get dense input: {end_time - start_time} seconds")

    await loop.run_in_executor(None, algorithm_state.mapping_input, input_text, dense_input)
    recipes = await data_handler.get_recipes(input_text)
    dense_all, dense_ingr = await loop.run_in_executor(None, data_handler.get_embeddings_recipe, recipes)
    await loop.run_in_executor(None, algorithm_state.mapping_output, recipes, dense_all, dense_ingr)
    await loop.run_in_executor(None, algorithm_state.rating_recipe, rating)

    html = await loop.run_in_executor(None, render_steps, input_text, algorithm_state)
    return html, algorithm_state


with gr.Blocks(js=javascript_code,
               css="""
    .step-images-container {
        display: flex;         /* Kunci utama: membuat item di dalamnya berbaris horizontal */
        flex-direction: row;   /* Mengatur arah baris (default) */
        flex-wrap: wrap;       /* Izinkan gambar pindah ke baris baru jika tidak muat */
        gap: 8px;              /* Memberi jarak antar gambar */
        margin-bottom: 12px;   /* Memberi jarak antara baris gambar dan teks di bawahnya */
    }

    /* Styling untuk setiap gambar individu */
    .step-image {
        height: 90px;          /* Atur tinggi gambar agar seragam */
        width: 120px;
        object-fit: cover;     /* Mencegah gambar menjadi gepeng/penyok */
        border-radius: 8px;    /* Membuat sudut gambar melengkung */
    }

    /* Sedikit penyesuaian pada step-item dan step-text */
    .step-item {
        margin-bottom: 20px; /* Jarak antar langkah */
    }
    .step-text {
        text-align: left;
    }

    .step-item img { border-radius: 8px; float: left; margin-right: 10px; }

    .carousel-container { position: relative; width: 100%; margin: auto; overflow: hidden; border: 1px solid #ddd; border-radius: 12px; height: 400px;}
    .carousel-slide { display: none; text-align: center; padding: 20px; }
    .cover-img { max-width: 100%; border-radius: 12px; }
    .main-scroll { max-height: 400px; overflow-y: auto; text-align: center; }
    .ingredients-scroll { max-height: 400px; overflow-y: auto; text-align: left; }
    .steps-scroll { max-height: 300px; overflow-y: auto; text-align: left; }
    ul { list-style-position: inside; text-align: left; }
    .prev-btn, .next-btn { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.5); color: white; border: none; padding: 8px 12px; cursor: pointer; border-radius: 50%; z-index: 10; }
    .prev-btn { left: 10px; }
    .next-btn { right: 10px; }
    """
               ) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Recipe Recommendation Here")
            uploader = gr.UploadButton("Upload a file", file_types=[
                                       "image"], file_count="single")
            preview = gr.Image(label="Uploaded Preview")
            user_preference = gr.Textbox(
                label="Type your prefereces here (Optional)", interactive=True)
            ingredients = gr.Textbox(
                label="Type your ingredients here...", interactive=True)
            generate_btn = gr.Button("Generate Recipe")
            # hidden dulu

        with gr.Column(scale=1
                       ):
            steps_html = gr.HTML(render_steps(),
                                 label="Steps will appear here")  # awalnya kosong
            rating = gr.Slider(
                minimum=-5, maximum=5, step=1,
                label="Rating For Recommendation",
                value=0, interactive=True, visible=False
            )
            next_btn = gr.Button("Next Recommendation", visible=False)

    uploader.upload(upload_file, uploader, preview)
    uploader.upload(text_replace, uploader, ingredients)
    # klik generate -> munculkan rating + next_btn
    algo_state = gr.State(AlgorithmClass())   # 👈 per-user state object
    generate_btn.click(
        generate_recipe,
        inputs=[ingredients, user_preference, algo_state],
        outputs=[steps_html, rating, next_btn, algo_state]
    )

    # klik next recommendation
    next_btn.click(
        next_recommendation,
        inputs=[ingredients, rating, algo_state],
        outputs=[steps_html, algo_state]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
