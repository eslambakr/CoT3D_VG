"""
Generate the real text prompt which will be fed to text-2-image models based on the fed meta-prompt.
"""
import openai
import datetime
from tqdm import tqdm
import csv
import sys
from csv import DictReader
  

def read_csv_in_dict(csv_path):
    with open(csv_path, 'r') as f:
        dict_reader = DictReader(f)
        list_of_dict = list(dict_reader)
    return list_of_dict


def run_chatgpt(model, temp, meta_prompt, max_tokens):
    # Define the parameters for the text generation
    completions = openai.Completion.create(engine=model, prompt=meta_prompt, max_tokens=max_tokens, n=1, stop=None,
                                           temperature=temp)
    gen_prompt = completions.choices[0].text.strip().lower()
    # Print the generated text
    print("The meta prompt is --> ", meta_prompt)
    print("ChatGPT output is --> ", gen_prompt)
    return gen_prompt


def save_lst_strings_to_txt(saving_txt, lst_str):
    file = open(saving_txt, 'w')
    for item in lst_str:
        file.write(item + "\n")
    file.close()


def save_prompts_in_csv(lst, saving_name):
    # Save output in csv:
    keys = lst[0].keys()
    with open(saving_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(lst)


def wait_one_n_mins(n_mins=1):
    endTime = datetime.datetime.now() + datetime.timedelta(minutes=n_mins)
    while True:
        if datetime.datetime.now() >= endTime:
            break


if __name__ == '__main__':
    # Set your API key
    openai.api_key = sys.argv[1]
    num_samples = int(sys.argv[2])  # e.g., 1000
    csv_path = sys.argv[3]
    max_tokens = 70

    # Read dataset
    nr3d_lst_dict = read_csv_in_dict(csv_path=csv_path)

    print(len(nr3d_lst_dict))
    print(nr3d_lst_dict[0])
    
    meta_prompt = "given the following sentence extract the objects only "
    meta_prompt = "given the following sentence give me the logical order of objects to reach to the target object:"
    #meta_prompt = "given the following sentence seperate relation between objects"
    #meta_prompt = "given the following sentence extract spatial relationship words between objects pairs and the objects itself, each object pairs and their relation in one line [object1] [object2] [relation]"
    #meta_prompt = "extract spatial relationship words between objects pairs and the objects themselves, in this format [object 1], [object 2]: [spatial relation word]. And each relation in seperate line "
    #meta_prompt = "extract in this format [object 1], [object 2]: [spatial relation word]. And each relation in seperate line "
    #meta_prompt = "extract all object pairs and their spatial relationship in this format [object 1], [object 2]: [spatial relation word]. And each relation in seperate line "
    #meta_prompt = "given the following sentence extrct subject, objects, and relations "
    meta_prompt = "given this relation 'closest to' tell me what is the new relation to swap the object and the subject."
    for i in tqdm(range(num_samples)):
        i = 5
        #gpt_in = meta_prompt + ' " ' + nr3d_lst_dict[i]["utterance"] + ' " '
        gpt_in = meta_prompt

        # Handle openAI timeout:
        chatgpt_out = None
        while chatgpt_out is None:
            try:
                chatgpt_out = run_chatgpt(model="text-davinci-003", temp=0, meta_prompt=gpt_in, max_tokens=max_tokens)
            except:
                print("OpenAI server out! Will try again Don't worry :D")
                pass

    """
        meta_prompt_dict.update({"synthetic_prompt": final_prompt})

        if (i % 20 == 0) and (i != 0) and chatgpt_out:
            wait_one_n_mins(n_mins=1)  # wait one minute to not exceed the openai limits

    generated_dict_lst = {k: [dic[k] for dic in generated_lst_dict] for k in generated_lst_dict[0]}

    # Saving:
    save_prompts_in_csv(lst=generated_lst_dict, saving_name="synthetic_" + skill + "_prompts.csv")
    if skill == "counting":
        save_lst_strings_to_txt(saving_txt="vanilla_" + skill + "_prompts.txt",
                                lst_str=generated_dict_lst['vanilla_prompt'])
    save_lst_strings_to_txt(saving_txt="meta_" + skill + "_prompts.txt", lst_str=generated_dict_lst['meta_prompt'])
    save_lst_strings_to_txt(saving_txt="synthetic_" + skill + "_prompts.txt",
                            lst_str=generated_dict_lst['synthetic_prompt'])

    print("Done")
    """