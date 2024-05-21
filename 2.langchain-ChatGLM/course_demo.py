# -*- coding : utf-8 -*-
# @Time     : 2023/7/21 - 15:35
# @Author   : 知行合一
# @FileName : course_demo.py
#

# 导入相关包
import os
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
from chains.local_doc_qa import LocalDocQA
from configs.model_config import (
    LLM_HISTORY_LEN,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    VECTOR_SEARCH_TOP_K,
    STREAMING,
)


# 是否显示资料出处
REPLY_WITH_SOURCE = True

def main():
    # 加载大语言模型
    llm_model_ins = shared.loaderLLM()  
     # 设置历史长度
    llm_model_ins.history_len = LLM_HISTORY_LEN 
    # 加载本地文档问答模型
    local_doc_qa = LocalDocQA()  
    # 配置，详细的见 ./config/model_config.py
    local_doc_qa.init_cfg(
        llm_model=llm_model_ins,            # 初始化的大语言模型，当前为 chatglm2-6b
        embedding_model=EMBEDDING_MODEL,    # Embedding 模型, 当前为"GanymedeNil/text2vec-large-chinese"
        embedding_device=EMBEDDING_DEVICE,  # 运行的设备, cuda or cpu
        top_k=VECTOR_SEARCH_TOP_K,          # 向量检索返回的结果个数，默认top5,
    )  

    # 向量数据库的路径
    vs_path = None 
    # 设置本地知识库的路径，会自动加载该路径下的所有文件
    filepath = "./knowledge_base/samples/content/"
    # 按照句子切分指定目录下的所有文件
    docs, loaded_files = local_doc_qa.load_and_split_file(filepath)

    # 构建或者加载 向量数据库
    temp_vs_path = local_doc_qa.create_or_add_vector_store(docs)

    if temp_vs_path is not None:
        # 如果加载成功，就将vs_path设置为加载的路径
        vs_path = temp_vs_path
        print(f"加载的vs_path为: {vs_path}")
    else:
        # 如果加载失败，就打印错误信息
        print("文件加载识别！请重新输入本地知识文件路径！")
        return None


    history = []
    while True:
        query = input("请输入问题：")
        last_print_len = 0
        # 获取答案
        for resp, history in local_doc_qa.get_knowledge_based_answer(query=query, vs_path=vs_path, chat_history=history, streaming=STREAMING): 
            # 如果是流式的，就不断的打印
            if STREAMING: 
                print(resp["result"][last_print_len:], end="", flush=True)
                last_print_len = len(resp["result"])
            else:
                # 打印一次
                print(resp["result"]) 
        # 是否显示资料出处
        if REPLY_WITH_SOURCE: 
            source_text = [
                f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                for inum, doc in enumerate(resp["source_documents"])
            ]
            print("\n\n" + "\n\n".join(source_text))


if __name__ == "__main__":
    args = None
    args = parser.parse_args()
    args_dict = vars(args)
    # 加载检查点
    shared.loaderCheckPoint = LoaderCheckPoint(args_dict) 
    # 启动
    main()
