import json
import re
import sys
from io import BytesIO

import requests
from dotenv import load_dotenv
from langchain import callbacks
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Call_Center_Operation_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Consulting_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Content_Production_Service_Contract_(Request_Form).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Customer_Referral_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Draft_Editing_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Graphic_Design_Production_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/software_development_outsourcing_contracts.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Technical_Verification_(PoC)_Contract.pdf",
]
# ==============================================================================


# ==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
# ==============================================================================


# PDFファイルからテキストを読み込む関数
def load_pdf(pdf_url: str) -> Document:
    response = requests.get(pdf_url)
    response.raise_for_status()

    with BytesIO(response.content) as file:
        reader = PdfReader(file)
        pdf_title = re.split(
            r"契約|契約書",
            re.sub(
                r"\s*1\s*",
                "",
                reader.pages[0].extract_text().replace("\n", "").replace(" ", ""),
            ),
            1,
        )[0]
        pdf_text = "".join(
            page.extract_text().replace("\n", "") for page in reader.pages
        )
        return Document(
            page_content=pdf_text,
            metadata={"title": pdf_title, "source": pdf_url},
        )


def retrieve_documents_by_title(title: str) -> list[Document]:
    documents = []
    for pdf_url in pdf_file_urls:
        documents.append(load_pdf(pdf_url))
    return [doc for doc in documents if doc.metadata["title"] == title]


# 回答を生成する関数
def generate_answer(question: str, context_text: str) -> str:
    openai = ChatOpenAI(model=model, temperature=0.0)
    prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context_text}\n\nQuestion: {question}"
    messages = [
        (
            "system",
            "あなたは優秀な法律の専門家です。与えられる契約資料に基づき、ユーザーの質問に丁寧かつわかりやすく回答してください。",
        ),
        ("user", prompt),
    ]
    response = openai.invoke(messages)
    return response.content


# RAGパイプラインの実装
def rag_implementation(question: str) -> str:
    # questionから「XX契約」「XX契約書」を抽出
    related_doc_titles = [
        title.replace("契約", "")
        for title in re.findall(r"(\S+契約)", question.replace(" ", ""))
    ]

    context_docs = []
    for title in related_doc_titles:
        context_docs.extend(retrieve_documents_by_title(title))

    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    answer = generate_answer(question, context_text)
    # print(answer)
    return answer


# ==============================================================================


# ==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
# ==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        run_id = cb.traced_runs[0].id

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
# ==============================================================================
