from moviepy.editor import VideoFileClip, TextClip, AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from diffusers import DiffusionPipeline
import scipy
from transformers import AutoProcessor, BarkModel
import nltk
import re
import openai
import rnm_text
import PIL
import os
import torch

PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
openai.api_key = os.environ['OPENAI_API_KEY']


def generate_audio(prompt, prompt_index, model, processor):
    voice_preset = "v2/en_speaker_6"
    inputs = processor(prompt, voice_preset=voice_preset).to(device())
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    audio_file = f"audio_{prompt_index}.wav"
    full_path = os.path.join(os.getcwd(), f"audios/{audio_file}")
    scipy.io.wavfile.write(full_path, rate=sample_rate, data=audio_array)
    return full_path


def generate_image(prompt, prompt_index, pipe):
    image = pipe(prompt).images[0]
    image_file = f"image_{prompt_index}.png"
    full_path = os.path.join(os.getcwd(), f"images/{image_file}")
    image.save(full_path)
    return full_path


def generate_video(prompt, prompt_index, audio_file, image_file):
    audio = AudioFileClip(audio_file)
    text = TextClip(prompt, fontsize=20, color='white', size=(640, 140), bg_color='black', method='caption', align='south')
    text = text.set_position(('center', 'bottom')).set_duration(audio.duration)
    image = ImageClip(image_file)
    image = image.resize(width=text.w)
    image = image.set_position(('center', 'center')).set_duration(audio.duration)
    clips = [image, text.set_audio(audio)]
    final_video = CompositeVideoClip(clips, size=(image.w, image.h + 140))
    video_file = f"video_{prompt_index}.mp4"
    full_path = os.path.join(os.getcwd(), f"videos/{video_file}")
    final_video.write_videofile(full_path, fps=60)
    return full_path


def rephrase(sentences):  # Rephrase sentences exceeding 220 characters (bark outputs 13s max duration audio) : ~270 characters ~= 13 seconds (speech is a bit too fast)
    sentences_array = []
    for prompt in sentences:
        if (len(prompt) > 220):
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"split in multiple sentences with maximum length of 220 characters per sentence the following text : {prompt}"}])
            split_sentence = nltk.sent_tokenize(completion.choices[0].message.content)
            sentences_array.extend(split_sentence)
            print(f"sentence {prompt} was split into {split_sentence}")
        else:
            sentences_array.append(prompt)
    return sentences_array


def text_to_videos(sentences_array, model, processor, pipe):
    videos_array = []
    for i, prompt in enumerate(sentences_array):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"converting sentence #{i} : {prompt}")
                audio_file = generate_audio(prompt, i, model, processor)
                image_file = generate_image(prompt, i, pipe)
                video_file = generate_video(prompt, i, audio_file, image_file)
                videos_array.append(VideoFileClip(video_file))
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    print(f"Retrying... (attempt {attempt}/{max_attempts})")
                else:
                    print("Max retry attempts reached. Operation failed.")
    return videos_array


def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def files_path(folder):
    folder_path = os.path.join(os.getcwd(), f"{folder}/")
    files_name = os.listdir(folder_path)
    return [folder_path + file for file in files_name]


def clean_media():
    images = files_path("images")
    audios = files_path("audios")
    videos = files_path("videos")
    files_list = images + audios + videos
    [os.remove(file) for file in files_list]


def main():
    model = BarkModel.from_pretrained("suno/bark")
    model = model.to(device())
    model = model.to_bettertransformer()
    model.enable_cpu_offload()
    processor = AutoProcessor.from_pretrained("suno/bark")

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, variant="fp16", use_safetensors=True).to(torch.device("cuda"))

    text_array = [rnm_text.text2(), rnm_text.text3(), rnm_text.text4(), rnm_text.text5(), rnm_text.text6(), rnm_text.text7(), rnm_text.text8(), rnm_text.text9(),
                  rnm_text.text10(), rnm_text.text11(), rnm_text.text12(), rnm_text.text13(), rnm_text.text14(), rnm_text.text15(), rnm_text.text16()]
    for i, text in enumerate(text_array):
        text = re.sub(r'\.(?=[^\s])', '. ', text)
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)
        print(f"{len(sentences)} sentences")
        sentences_array = rephrase(sentences)
        print(f"{len(sentences_array)} split sentences to translate to speech")
        videos_array = text_to_videos(sentences_array, model, processor, pipe)
        final_video = concatenate_videoclips(videos_array, method='compose')
        final_video_path = os.path.join(os.getcwd(), f"compilation_videos/compilation_output{i+2}.mp4")
        final_video.to_videofile(final_video_path, fps=60)
        clean_media()


if __name__ == "__main__":
    main()