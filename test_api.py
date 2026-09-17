import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"

if not API_KEY:
    print("错误：未找到 DEEPSEEK_API_KEY 环境变量")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def ask_deepseek(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": question}],
            timeout=30,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"请求失败：{e}"

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else input("请输入问题：")
    print(f"\n回答：{ask_deepseek(q)}")