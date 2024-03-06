import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


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


def to_keywords(input_string):
    """ Given an input text string, return a list of keywords (supports English only)
    :param input_string:
    :return:
    """
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


def split_text(paragraphs, chunk_size=300, overlap_size=100):
    """ 按指定 chunk_size 和 overlap_size 交叠割文本，避免缺失上下文导致的向量不匹配
    :param paragraphs:
    :param chunk_size:
    :param overlap_size:
    :return:
    """
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev_len = 0
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
