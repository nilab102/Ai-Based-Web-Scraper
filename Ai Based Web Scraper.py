#Please configure your environment to utilize the API key provided for OpenAI or Anthropic.
#pip install langchain-anthropic
#!pip install -q langchain-openai langchain playwright beautifulsoup4 playwright install

# Set env var OPENAI_API_KEY or load from a .env file:
# import dotenv
# dotenv.load_dotenv()
import dotenv
dotenv.load_dotenv()

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# Load HTML
loader = AsyncChromiumLoader(["https://www.wsj.com"])
html = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["span"])

# Result
docs_transformed[0].page_content[0:500]

from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

from langchain_community.document_transformers import Html2TextTransformer

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
docs_transformed[0].page_content[0:500]

#from langchain_openai import ChatOpenAI

#llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

from langchain_anthropic import ChatAnthropic
#model = ChatAnthropic(model='claude-3-opus-20240229')
#llm = ChatAnthropic(temperature=0, model="claude-3-opus-20240229")
llm = ChatAnthropic (model="claude-1", temperature=0, max_tokens_to_sample=2000)

from langchain.chains import create_extraction_chain



def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)

import pprint

from langchain_text_splitters import RecursiveCharacterTextSplitter


def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["span"]
    )
    print("Extracting content with LLM")

    # Print the transformed documents
    for doc in docs_transformed:
        print("Document Content:")
        print(doc.page_content)

    # Split documents if they are sufficiently long
    if docs_transformed:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )
        splits = splitter.split_documents(docs_transformed)
        
        if splits:
            # Process the first split
            extracted_content = extract(schema=schema, content=splits[0].page_content)
            pprint.pprint(extracted_content)
            return extracted_content
        else:
            print("No content splits found")
            return None
    else:
        print("No documents transformed")
        return None





def create_schema_template():
    properties = {}
    required_properties = []

    while True:
        property_input = input("Enter a property name and type (e.g., 'X M'), or 'q' to finish: ")
        if property_input.lower() == 'q':
            break

        try:
            name, prop_type = property_input.split()
        except ValueError:
            print("Invalid input format. Please enter the property name and type separated by a space.")
            continue

        properties[name] = {"type": prop_type}
        required_properties.append(name)

    required_input = input("Enter the required properties (comma-separated): ")
    required_properties = [prop.strip() for prop in required_input.split(",")]

    schema = {
        "properties": properties,
        "required": required_properties
    }

    return schema

if __name__ == "__main__":
    schema_template = create_schema_template()
    import json
    print(json.dumps(schema_template, indent=2))

def get_urls():
    urls = []
    while True:
        url = input("Enter a URL (or 'q' to finish): ")
        if url.lower() == 'q':
            break
        urls.append(url)
    return urls

schema = create_schema_template()
user_urls = get_urls()
#urls = ["https://www.amazon.in/s?k=poco"]
extracted_content = scrape_with_playwright(urls, schema=schema)



