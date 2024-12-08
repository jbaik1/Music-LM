# Music Language Model

Music is often considered as a universal language. That is, every culture has developed their own kind of music, and we eventually learn to enjoy various genres, even if some of them feel foreign at first.

Why not try modeling music as another type of language? 

A lot of research has already been done for this topic, and for this project I used a lot of great tools that others have made available:

The [Maestro](https://magenta.tensorflow.org/datasets/maestro) dataset 

The [Miditok](https://github.com/Natooz/MidiTok) python package

[nanoGPT](https://github.com/karpathy/nanoGPT) to train a new model without having to recreate a GPT model from scratch.

The Maestro dataset contains classical western repertoire from the 17th to early 20th century. It has around 1276 pieces.

Miditok tokenizes music, considering notes characterized by their pitch, duration, and volume. It can be thought of as tokenizing chords as “words”. Miditok uses BPE, as well as the TSD algorithm. 
Data augmentation performed since there isn't that much music compared to text. 
The exact configuration for data augmentation (as implemented in Miditok) is:

```
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 16,
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,
    "num_tempos": 16,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
```

Once the tokenization is complete, a new model is created using nanoGPT, and is trained using the Maestro dataset. The training config file is included here.

The training process used an NVIDIA Tesla V100, using Google’s cloud platform (renting a gpu server with free credits). Given the maximum GPU memory of around 15GB, the size of the transformer model is relatively small (only 1 GPU was allowed for my server). 

The model is stored in huggingface under [jwb23/music-lm](https://huggingface.co/jwb23/music-lm/tree/main).

After training for about 24 hours, some samples are included in this github repo.
The sound quality can vary depending on what software you are using to open the midi files, but the notes are the same.

Even with significant limitations on the size of the model as well as the size of the dataset, this sounds surprisingly good! 

#### Is there only one language for music?

The Maestro dataset only has music from classical western music. However, even within this genre there are significant differences. Baroque, classical, romantic, and modern classical music don't really sound the same. An extreme example is comparing Mozart's Eine Kleine Nachtmusik with Schoenberg's atonal pieces. To be fair, it's also true that all of these eras do share a common understanding of western harmony (except atonal music).

Now there's the issue of combining multiple genres. Jazz and pop songs aren't so bad, since they still have your typical western chord progressions. But what about non-western music([world music](https://en.wikipedia.org/wiki/World_music))? 
If you include every genre, then it might be like training your language model on English and French at the same time. The model might just output random notes.

Also, how would you deal with different "instrumentations"? You can't really represent rap or drums with music notes.  

#### What would pre-trained models look like?

Given a large amount of data for classic western music, it might be possible to create some sort of a “pre-trained” model. When fine-tuning it for a specific composer, the model may be able to generate music more in the style of that composer. 

#### Music Style Transfer 
Style transfer has been studied in vision and generative AI. An analogous subject within NLP is text style transfer. Given that we are able to tokenize music and train ML models as a result, this may enable us to do some version of “music style transfer”. For example, playing happy birthday in the style of Bach. 


