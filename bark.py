from moviepy.editor import VideoFileClip, TextClip, AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from diffusers import DiffusionPipeline
import scipy
from transformers import AutoProcessor, BarkModel
import nltk
import re
import torch
import openai
import rnm_text
import PIL
import os
import string
import ipdb

PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
openai.api_key = os.environ['OPENAI_API_KEY']


def generate_audio(prompt, prompt_index, model, processor):
    voice_preset = "v2/en_speaker_6"
    inputs = processor(prompt, voice_preset=voice_preset)
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
    text = TextClip(prompt, fontsize=25, color='white', size=(640, 140), bg_color='black', method='caption', align='south')
    text = text.set_position(('center', 'bottom')).set_duration(audio.duration)
    image = ImageClip(image_file)
    image = image.resize(width=text.w)
    image = image.set_position(('center', 'center')).set_duration(audio.duration)
    clips = [image, text.set_audio(audio)]
    final_video = CompositeVideoClip(clips, size=(image.w, image.h + 140))
    video_file = f"video_{prompt_index}.mp4"
    full_path = os.path.join(os.getcwd(), f"videos/{video_file}")
    final_video.write_videofile(full_path, fps=60)
    for media in [audio, image, final_video]:
        media.close()
    return full_path


def rephrase(sentences):  # Rephrase sentences exceeding 220 characters (bark outputs 13s max duration audio) : ~270 characters ~= 13 seconds (speech is a bit too fast)
    sentences_array = []
    for prompt in sentences:
        if (len(prompt) > 220):
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": f"split in multiple sentences with maximum length of 220 characters per sentence the following text : {prompt}"}])
                    split_sentence = nltk.sent_tokenize(completion.choices[0].message.content)
                    sentences_array.extend(split_sentence)
                    print(f"sentence {prompt} was split into {split_sentence}")
                    break
                except Exception as e:
                    print(f"Openai Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        print(f"Openai Retrying... (attempt {attempt}/{max_attempts})")
                    else:
                        print("Openai Max retry attempts reached. Operation failed.")
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


def bark_sentences(text):
    words = nltk.word_tokenize(text)
    sentences = []
    sentence = ""
    for i, word in enumerate(words):
        if (len(sentence) + len(word) < 220):
            sentence = f"{sentence}{word}" if (word in (string.punctuation + "’")) else f"{sentence} {word}"
            if (i == len(words)-1):
                sentences.append(sentence)
        else:
            sentences.append(sentence)
            sentence = word
    return sentences


def main():
    # Text to Speech model
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark")
    model = model.to_bettertransformer()

    # Text to Image model
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, variant="fp16", use_safetensors=True)

    text_array = [rnm_text.texttest()]
    for i, text in enumerate(text_array):
        # Text Tokenization
        text = re.sub(r'\.(?=[^\s])', '. ', text)
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)
        print(f"{len(sentences)} sentences")
        sentences_array = []
        for sentence in sentences:
            sentences_array.extend(bark_sentences(sentence))
        print(f"{len(sentences_array)} split sentences to translate to speech")
        # ipdb.set_trace(context=5)
        videos_array = text_to_videos(sentences_array, model, processor, pipe)
        final_video = concatenate_videoclips(videos_array, method='compose')
        final_video_path = os.path.join(os.getcwd(), f"compilation_videos/compilation_output{i}.mp4")
        final_video.to_videofile(final_video_path, fps=60)
        clean_media()


if __name__ == "__main__":
    main()
