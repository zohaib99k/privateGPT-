#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
from transformers import MarianMTModel, MarianTokenizer

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS


def translate_en_ar(text):
    # Load the pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-ar"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # Translate the text
    translation = model.generate(inputs, max_length=128)
    # Decode and return the translated text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text


def translate_ar_en(text):
    # Load the pre-trained translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-ar-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # Translate the text
    translation = model.generate(inputs, max_length=128)
    # Decode and return the translated text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text
def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nأدخل استعلامًا: ")
        if query == "exit":
            break

        translated_text = translate_ar_en(query)
        res = qa(translated_text)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        ans = translate_en_ar(answer)
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(ans)

        # # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()