import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
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


def retrieve_documents(question: str) -> list[Document]:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_pdf, pdf_file_urls))
    titles = [doc.metadata["title"] for doc in results]

    openai = ChatOpenAI(model=model, temperature=0.0)
    messages = [
        (
            "system",
            (
                "あなたは法律の専門家です。質問に関連する契約書のタイトルを候補の中から全て選び、リストで出力してください。"
                "出力は、リストのタイトルを','で区切ったものを出力してください。\n"
                "契約書1,契約書2,契約書3"
            ),
        ),
        (
            "user",
            (
                f"【質問】{question}\n"
                "【契約書のタイトルの候補】\n"
                f"{','.join(titles)}"
            ),
        ),
    ]
    related_doc_titles = openai.invoke(messages).content
    related_doc_titles = related_doc_titles.split(",")
    return [doc for doc in results if doc.metadata["title"] in related_doc_titles]


# 回答を生成する関数
def generate_answer(question: str, context_text: str) -> str:
    openai = ChatOpenAI(model=model, temperature=0.0)
    prompt = f"【コンテキスト】\n{context_text}\n\n【質問】{question}"
    messages = [
        (
            "system",
            (
                "あなたは法律の専門家です。"
                "ユーザーの質問には、可能な限り具体的かつ簡潔・端的な回答を、ユーザーが理解しやすいテキスト形式で提供してください。（マークダウン形式にしないこと！）"
                "また、与えられたコンテキストのみでは、生成する回答の正確性に不安がある場合はその旨を明示してください。コンテキストにない情報については、勝手に補ってはいけません。"
                "あなたの回答は、correctness, helpfulness, conciseness, harmlessnessで評価されることを覚えておいてください。\n"
                "回答の例を以下に示します。\n"
                "-----\n"
                "【質問】ソフトウェア開発業務委託契約について、委託料の金額はいくらですか？\n"
                "【回答】委託料の金額は金五百万円（税別）です。\n"
                "-----\n"
                "【質問】コールセンター業務委託契約における請求書の発行プロセスについて、締め日と発行期限を具体的に説明してください。\n"
                "【回答】受託者は毎月末日に締め、翌月5日までに請求書を発行する。\n"
                "-----\n"
            ),
        ),
        ("user", prompt),
    ]
    response = openai.invoke(messages)
    return response.content


# RAGパイプラインの実装
def rag_implementation(question: str) -> str:
    context_docs = retrieve_documents(question)
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
