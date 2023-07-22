MoE 4-bit Llama-v2 w/ QLora ELM Forest (BTM)

We are creating an MoE model by stitching together a bunch of QLora domain-specific adaptations on the base Llama v2 models. Each example during the inference would be routed through a set of QLora modules or their combination. This routing the main things that need to be learned in some way or the other. The goal is to solve an existing task better by using specialized domain experts. This is kind of an opensource and efficient attempt to mimic GPT4 with opensource Llama2 model.

Working document: https://docs.google.com/document/d/1YKDRCu7M9mflWrxKc1HeFs2HBWk4HXVrrOsLrIn6EXM/edit?usp=sharing
Discord Server: https://discord.gg/CZAJcWTZxX

**Setup**
  pip install -r requirements.txt

## Access to the Llama 2 weights

You will need to sign in with your HF account and sign the terms. You can find the terms in one of the model pages for Llama 2, e.g. here: https://huggingface.co/meta-llama/Llama-2-70b-hf

Then, you will want to [log in to HF with a token](https://huggingface.co/docs/huggingface_hub/quick-start#login):
```
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

## Finetuner Setup

```
python3 main.py --finetune --config <path to config>
```

### Modes
The Finetuner can be run in two different modes: `qlora` and `lora`.

#### qlora Mode
To run the Finetuner in `qlora` mode, use the following command:
```
python3 main.py --finetune --mode qlora --config <path to config>
```

#### lora Mode
To run the Finetuner in `lora` mode, use the following command:
```
python3 main.py --finetune --mode lora --config <path to config>
```
 