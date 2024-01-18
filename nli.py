import argparse
import json
import os
import openai
import random
from tqdm import tqdm
from time import sleep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--example", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--example_num", type=int, default=-1)
    parser.add_argument("--learning", type=str, choices=["zero-shot", "few-shot", "few-shot_explanation", "few-shot_humans_discussion", "few-shot_pseudo_discussion"])
    parser.add_argument("--task", type=str, choices=["snli", "anli1", "anli2", "anli3"])
    args = parser.parse_args()

    return args


def set_gpt():

    API_KEY = "Your API key"
    openai.api_key = API_KEY


def create_dataset(args, examples):
    prompt_batchs = []
    input_batchs = []
    prompts = []
    inputs = []

    task_description = "Please select the label whether the premise and hypothesis are entailment, contradiction, or neutral. "

    with open(args.input) as f:
        for l in f:
            d = json.loads(l)
            if args.task == "snli":
                gold_label = d["gold_label"]
                annotator_labels = d["annotator_labels"]
                premise = d["sentence1"]
                hypothesis = d["sentence2"]
            elif args.task == "anli1" or args.task == "anli2" or args.task == "anli3":
                if d["label"] == "e":
                    gold_label = "entailment"
                elif d["label"] == "c":
                    gold_label = "contradiction"
                elif d["label"] == "n":
                    gold_label = "neutral"
                premise = d["context"]
                hypothesis = d["hypothesis"]

            prompt = task_description 
            
            if args.learning == "zero-shot":
               pass
            else:
                for i, example in enumerate(examples):
                    example_premise = example["premise"]
                    example_hypothesis = example["hypothesis"]
                    human_label = example["human_label"]
                    example_gold_label = example["gold_label"]
                    humans_discussion = ' '.join(example["humans_discussion"])
                    pseudo_discussion = example["humans_discussion"]

                    if args.learning == "few-shot":
                        prompt += f"\nPremise: {example_premise} Hypothesis: {example_hypothesis} Label: {example_gold_label} "
                    elif args.learning == "few-shot_explanation":
                        prompt += f"\nPremise: {example_premise} Hypothesis: {example_hypothesis} Label: {example_gold_label} Explanation: {explanation} "
                    elif args.learning == "few-shot_humans_discussion":
                        prompt += f"\nPremise: {example_premise} Hypothesis: {example_hypothesis} Label: {human_label} Discussion: {humans_discussion} "
                    elif args.learning == "few-shot_pseudo_discussion":
                        prompt += f"\nPremise: {example_premise} Hypothesis: {example_hypothesis} Label: {human_label} Discussion: {pseudo_discussion} "
                    if args.example_num <= i + 1:
                        break
            prompt += f"\nPremise: {premise} Hypothesis: {hypothesis} Label: "
            if args.task == "snli":
                input = [gold_label, premise, hypothesis, annotator_labels]
            elif args.task == "anli1" or args.task == "anli2" or args.task == "anli3":
                input = [gold_label, premise, hypothesis]

            inputs.append(input)
            prompts.append(prompt)
            if len(prompts) == args.batch_size:
                prompt_batchs.append(prompts)
                input_batchs.append(inputs)
                prompts = []
                inputs = []
    
    if len(prompts) > 0:
        prompt_batchs.appned(prompts)
        input_batchs.appned(inputs)

    return prompt_batchs, input_batchs


def main(args):

    random.seed(0)

    data_num = 0
    correct_num = 0
    ignored_data_count = 0
    saved_data_path = f"output/input_{args.task}_{args.learning}.json"

    with open(args.example) as f:
        examples = json.load(f)

    if os.path.exists(args.output):
        with open(saved_data_path) as f:
            batchs = json.load(f)
        with open(args.output) as f:
            results = json.load(f)
        if len(batchs[0]) == 0:
            print("End")
            exit()
    else:
        prompt_batchs, input_batchs = create_dataset(args, examples)
        batchs = [prompt_batchs, input_batchs]
        with open(saved_data_path, 'w') as fw:
            json.dump(batchs, fw, indent=4)
        results = []

    set_gpt()
    prompt_batchs, input_batchs = batchs

    for batch_idx in tqdm(range(len(prompt_batchs))):
        prompt_batch = prompt_batchs[batch_idx]
        input_batch = input_batchs[batch_idx]
        response = openai.Completion.create(engine="text-davinci-003",
                                            prompt=prompt_batch,
                                            max_tokens=3,
                                            temperature=0,
                                            logprobs=0,)

        for i in range(args.batch_size):
            gold_label = input_batch[i][0]
            premise = input_batch[i][1]
            hypothesis = input_batch[i][2]
            if args.task == "snli":
                annotator_labels = input_batch[i][3]
            prompt = prompt_batch[i]

            if gold_label == "-":
                ignored_data_count += 1
                continue
            data_num += 1

            system_output = response["choices"][i]["text"].lower().strip()
            system_label = system_output.split()[0]
            tokens = response["choices"][i]["logprobs"]["tokens"]
            token_logprobs = response["choices"][i]["logprobs"]["token_logprobs"]

            tokens_and_logprobs = []
            for token, token_logprob in zip(tokens, token_logprobs):
                if token == "<|endoftext|>":
                    break
                if token != "\n":
                    tokens_and_logprobs.append([token, token_logprob])
            sequence_avg_logprob = sum([token_and_logprob[1] for token_and_logprob in tokens_and_logprobs]) / len(tokens_and_logprobs)
            
            if gold_label == system_label:
                correct_num += 1
            
            if args.task == "snli":
                results.append({"premise": premise, "hypothesis": hypothesis, "prompt": prompt,
                                "system_label": system_label, "tokens_and_logprobs": tokens_and_logprobs,
                                "sequence_avg_logprob": sequence_avg_logprob,"gold_label": gold_label,
                                "system_output": system_output, "system_label": system_label, "annotator_labels": annotator_labels})
            else:
                results.append({"premise": premise, "hypothesis": hypothesis, "prompt": prompt,
                                "system_label": system_label, "tokens_and_logprobs": tokens_and_logprobs,
                                "sequence_avg_logprob": sequence_avg_logprob,"gold_label": gold_label,
                                "system_output": system_output, "system_label": system_label})
        
        print(f"Acc: {correct_num / data_num * 100:.2f}% \r", end="")
        with open(args.output, 'w') as fw:
            json.dump(results, fw, indent=4)
        with open(saved_data_path, 'w') as fw:
            json.dump([prompt_batchs[batch_idx+1:], input_batchs[batch_idx+1:]], fw, indent=4)
        sleep(5)

    with open(args.output, 'w') as fw:
        json.dump(results, fw, indent=4)
    with open(saved_data_path, 'w') as fw:
            json.dump([prompt_batchs[batch_idx+1:], input_batchs[batch_idx+1:]], fw, indent=4)
    print("Acc:", correct_num / data_num * 100)
    print("Data num:", data_num)
    print("Correct data num:", correct_num)
    print("Ignore data num:", ignored_data_count)


if __name__ == "__main__":
    args = parse_args()
    main(args)
