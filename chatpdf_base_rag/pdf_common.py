import re

import jieba
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize  # 断句
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


# 文本处理方法库
# nltk.download('punkt')  # 英文切词、词根、切句等方法
# nltk.download('stopwords')  # 英文停用词库


class PdfUtils:
    def __init__(self, filename):
        self.filename = filename
        pass

    def extract_text_from_pdf(self, select_page_numbers=None, min_line_length=1):
        """ 从 PDF 文件中（按指定页码）提取文字
        :param select_page_numbers: 指定要提取的的页
        :param min_line_length: 一行不超过 min_line_length 个单词，认为该段结束
        :return:
        """
        para_graphs = []
        buffer = ''
        full_text = ''
        # 按页提取全部文本
        for i, page_layout in enumerate(extract_pages(self.filename)):
            # 如果指定了页码范围，跳过范围外的页
            if select_page_numbers is not None and i not in select_page_numbers:
                continue
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    full_text += element.get_text() + '\n'
                # else:
                #     print(f'ignore {element}')

        # 按空行分隔，将文本重新组织成段落
        lines = full_text.split('\n')
        for text in lines:
            if len(text) >= min_line_length:
                buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
            elif buffer:
                para_graphs.append(buffer)
                buffer = ''
        if buffer:
            para_graphs.append(buffer)
        return para_graphs


def sent_tokenize_chinese(input_string):
    """按标点断句"""
    # 按标点切分
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # 去掉空字符串
    return [sentence for sentence in sentences if sentence.strip()]


def to_keywords(input_string, is_English=True):
    """ Given an input text string, return a list of keywords (supports English only)
    :param input_string:
    :param is_English:
    :return:
    """
    if not is_English:
        return to_keywords_chinese(input_string)
    # Replace all non-alphanumeric characters with spaces using regular expressions
    no_symbols = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string)
    word_tokens = word_tokenize(no_symbols)
    # Load the stop word list
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    # Stem and filter out stop words
    filtered_sentence = [ps.stem(w)
                         for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)


def to_keywords_chinese(input_string):
    """将句子转成检索关键词序列"""
    # 按搜索引擎模式分词
    word_tokens = jieba.cut_for_search(input_string)
    # 加载停用词表
    stop_words = set(stopwords.words('chinese'))
    # 去除停用词
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)


def split_text(paragraphs, chunk_size=300, overlap_size=100, is_English=True):
    """ 将段落切割成句子，单条句子控制长度在chunk_size，允许向前有一定的 overlap，避免缺失上下文导致的向量不匹配
    :param is_English:
    :param paragraphs:
    :param chunk_size:
    :param overlap_size:
    :return:
    """
    if is_English:
        sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    else:
        sentences = [s.strip() for p in paragraphs for s in sent_tokenize_chinese(p)]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev = i - 1
        # 向前计算重叠部分
        while prev >= 0 and len(sentences[prev]) + len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap + chunk
        next = i + 1
        # 向后计算当前chunk
        while next < len(sentences) and len(sentences[next]) + len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    return chunks


def test():
    print(to_keywords_chinese("小明硕士毕业于中国科学院计算所，后在日本京都大学深造"))
    print(sent_tokenize_chinese("这是，第一句。这是第二句吗？是的！啊"))
