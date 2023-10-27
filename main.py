import os
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import torch
import random
from transformers import pipeline, set_seed
import time
from test_cases import return_cases


set_bits = 4
pretrained_model_dir = "gpt2"
if set_bits != 4:
    quantized_model_dir = pretrained_model_dir+"GPTQ"+f"{set_bits}"
else:
    quantized_model_dir = pretrained_model_dir+"GPTQ"


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)

quantize_config = BaseQuantizeConfig(

    bits=set_bits,
    group_size=-1,
    damp_percent=0.01,
    desc_act=False,
)



def get_max_seq_len():


    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
    model_config = model.config.to_dict()
    seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
    if any([k in model_config for k in seq_len_keys]):
        for key in seq_len_keys:
            if key in model_config:
                model.seqlen = model_config[key]
                break

    else:
        print("can't get model's sequence length from model config, will set to 2048.")

        model.seqlen = 2048

def quantize():


    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
    get_max_seq_len()

    seqlen = model.seqlen


    n_samples = 1024
    data = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split=f"train[:{n_samples*5}]")

    tokenized_data = tokenizer("\n\n".join(data['text']), return_tensors='pt')



    examples = []

    for _ in range(n_samples):

        i = random.randint(0, tokenized_data.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        input_ids = tokenized_data.input_ids[:, i:j]
        attention_mask = torch.ones_like(input_ids)
        examples.append({'input_ids': input_ids, 'attention_mask': attention_mask})


    model.quantize(examples)

    # save quantized model
    model.save_quantized(quantized_model_dir)  


    repo_id = f"pavfi-at-m/{quantized_model_dir}"  
    commit_message = f"GPT2{pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"

    model.push_to_hub(repo_id, 
                      save_dir=quantized_model_dir, 
                      use_auth_token=True, 
                      commit_message=commit_message)
    tokenizer.push_to_hub(repo_id)


def main():

    quantize()
    tests = return_cases()
    out = []

    # load quantized model to the first GPU
    quantized_model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir)
    
    
    generator = pipeline('text-generation', model=pretrained_model_dir)

    for  test in tests:
        out.append(testing(test,quantized_model, generator))



    case = 1
    for result in out:
        print(case)
        case+=1
        print(f"Quant model result: {result[0]}")
        print(f"Normal model result: {result[1]}")
        print(f"Speed change result: {result[2]}")
        print("\n" * 2)



    average=0
    quant_time = 0
    nor_time = 0 

    for i in range(0,len(out)):
        average+=out[i][2] 
        quant_time+=out[i][4]
        nor_time+=out[i][3]

    average = average/(len(out))

    quant_time = quant_time/len(out)
    norm_time = nor_time/len(out)
    
    print(f"Average percentage speed change: {average} quant: {quant_time} norm: {norm_time}")


def testing(test_string, quantized_model,generator):

    max_len = random.randint(10,40)



    start = time.time()
    quant = (tokenizer.decode(quantized_model.generate(**tokenizer(test_string, return_tensors="pt", max_length=max_len, truncation=True).to(quantized_model.device))[0]))
    end = time.time()
    quant_time = end-start
    print(f"Quant {quant_time} ")

    set_seed(42)
    start= time.time()
    nor = (generator(test_string, max_length=max_len))

    end = time.time()

    nor_time = end-start
    print(f"Norm {nor_time} \n\n\n")


    percentage_change = (nor_time-quant_time)/nor_time * 100
    return [quant,nor, percentage_change, nor_time, quant_time]



if __name__ == "__main__":

    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()



