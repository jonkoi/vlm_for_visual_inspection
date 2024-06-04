# Vision Language Models for Visual Inspection

This project is for the completion of the Special curriculum FYS-8805 Generative AI. The topic of this project is the implementation of vision-language models to perform visual question answering on two tasks: Insulator disc counting and pole section matching.
## Installation

1. Clone this repo

```bash
git clone https://github.com/jonkoi/vlm_for_visual_inspection.git
cd vlm_for_visual_inspection
```

2. Create conda environment

```Shell
conda create -n vlm_inspect python=3.9 -y
conda activate vlm_inspect
```

3. Install dependencies

```shell
pip install -r requirements.txt
```

4. Download models and place them in current pwd

Pole section matching model with the following link: [Pole section matching model](https://drive.google.com/drive/folders/1r5V7M_2KUKGcPh9tJ_J0rwfQn82hbC4B?usp=sharing)

Default MiniCPM-V-2 model at: [MiniCPM-V-2](https://huggingface.co/openbmb/MiniCPM-V-2)

MiniCPM-V-2 model fine-tuned on Insulator disc counting at: [MiniCPM-V-2 Insulator disc counting](https://drive.google.com/drive/folders/1Fz16KGa8N2SFz3mxJVlFoIwW0Y5w5XkZ?usp=sharing)

Idefics2 model fine-tuned on Insulator disc counting at: [Idefics2 Insulator disc counting](https://drive.google.com/drive/folders/1ivA7diNCfvnT9nti39goLkHbKFr4IrrO?usp=sharing)

## Demo

To run the demo for Insulator disc counting, run the following command:

```bash
python ins_count_web.py.py
```

To run the demo for Pole section matching, run the following command:

```bash
python polesec_matching_web.py
```
