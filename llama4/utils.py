import os
import requests
import tiktoken
import html2text
from openai import OpenAI
from bs4 import BeautifulSoup

def get_llama_client():
    client = OpenAI(
        api_key=os.getenv("LLAMA_API_KEY"),
        base_url=os.getenv("LLAMA_BASE_URL"),
    )
    return client

def scrape_cuda_ptx_docs():
    url = "https://docs.nvidia.com/cuda/parallel-thread-execution/"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    article_body = soup.find('div', {'itemprop': 'articleBody'})
    if not article_body:
        raise ValueError("Could not find div with itemprop='articleBody'")
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    markdown_content = h.handle(str(article_body))
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(markdown_content)
    token_count = len(tokens)
    return markdown_content, token_count


def get_cuda_best_practices():
    cuda_best_practices_url = "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/"
    response = requests.get(cuda_best_practices_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    article_body = soup.find('div', {'itemprop': 'articleBody'})
    if not article_body:
        raise ValueError("Could not find div with itemprop='articleBody'")
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    markdown_content = h.handle(str(article_body))
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(markdown_content)
    token_count = len(tokens)
    return markdown_content, token_count



if __name__ == "__main__":
    markdown_content, token_count = get_cuda_best_practices()
    print("First 100 characters:")
    print(markdown_content[:100])
    print("\nLast 100 characters:")
    print(markdown_content[-100:])