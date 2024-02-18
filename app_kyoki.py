import MeCab
import pandas as pd
import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from itertools import combinations
import io
import chardet  # エンコーディング検出用

# ページのレイアウトを設定
st.set_page_config(page_title="テキスト可視化", layout="wide", initial_sidebar_state="expanded")

# フォントのパスを指定
font_path = "ipaexg.ttf"

# タイトルの設定
st.title("テキスト可視化")

# サイドバーにアップロードファイルのウィジェットを表示
st.sidebar.markdown("# ファイルアップロード")
uploaded_file = st.sidebar.file_uploader("テキストファイルをアップロードしてください", type="txt")

# MeCabのインスタンスを作成
tagger = MeCab.Tagger()

# ワードクラウド生成関数
def generate_wordcloud(text, font_path):
        # ワードクラウドの処理
        st.markdown("## ワードクラウド")
        pos_options = ["名詞", "形容詞", "動詞", "副詞", "助詞", "助動詞", "接続詞", "感動詞", "連体詞", "記号", "未知語"]
        # マルチセレクトボックス
        selected_pos = st.sidebar.multiselect("品詞選択", pos_options, default=["名詞"])
        if st.sidebar.button("生成"):
            with st.spinner("Generating..."):
                node = tagger.parseToNode(text)
                words = []
                while node:
                    if node.surface.strip() != "":
                        word_type = node.feature.split(",")[0]
                        if word_type in selected_pos: # 対象外の品詞はスキップ
                            words.append(node.surface)
                    node = node.next
                word_count = Counter(words)
                wc = WordCloud(
                    width=800,
                    height=800,
                    background_color="white",
                    font_path=font_path, 
                )
                # ワードクラウドを作成
                wc.generate_from_frequencies(word_count)
                # ワードクラウドを表示
                st.image(wc.to_array())

# 出現頻度表生成関数
def generate_frequency_table(text):
        # 出現頻度表の処理
        st.markdown("## 出現頻度表")
        pos_options = ["名詞", "形容詞", "動詞", "副詞", "助詞", "助動詞", "接続詞", "感動詞", "連体詞", "記号", "未知語"]
        # マルチセレクトボックス
        selected_pos = st.sidebar.multiselect("品詞選択", pos_options, default=pos_options)
        if st.sidebar.button("生成"):
            with st.spinner("Generating..."):
                node = tagger.parseToNode(text)
                # 品詞ごとに出現単語と出現回数をカウント
                pos_word_count_dict = {}
                while node:
                    pos = node.feature.split(",")[0]
                    if pos in selected_pos:
                        if pos not in pos_word_count_dict:
                            pos_word_count_dict[pos] = {}
                        if node.surface.strip() != "":
                            word = node.surface
                            if word not in pos_word_count_dict[pos]:
                                pos_word_count_dict[pos][word] = 1
                            else:
                                pos_word_count_dict[pos][word] += 1
                    node = node.next

                # カウント結果を表にまとめる
                pos_dfs = []
                for pos in selected_pos:
                    if pos in pos_word_count_dict:
                        df = pd.DataFrame.from_dict(pos_word_count_dict[pos], orient="index", columns=["出現回数"])
                        df.index.name = "出現単語"
                        df = df.sort_values("出現回数", ascending=False)
                        pos_dfs.append((pos, df))

                # 表を表示
                for pos, df in pos_dfs:
                    st.write(f"【{pos}】")
                    st.dataframe(df, 400, 400)

# 共起ネットワーク生成関数
def generate_cooccurrence_network(text, font_path):
        # 共起ネットワークの処理
        st.markdown("## 共起ネットワーク")
        # 共起ネットワークを描画する関数
        def draw_cooccurrence_network(G, ax=None):
          pos = nx.spring_layout(G, k=0.5, iterations=20)
          nx.draw(G, pos, with_labels=True, node_size=5000, node_color='skyblue', edge_color='gray',
             linewidths=0.5, font_size=10, ax=ax, font_family=font_prop.get_name())  # font_familyを追加
          edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
          nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', ax=ax)
        # フォントプロパティを設定
        font_prop = font_manager.FontProperties(fname=font_path)

        print('step1 pass')
 
        if st.sidebar.button("生成"):
          with st.spinner("Generating..."):
            from collections import defaultdict

            # MeCabでテキストを形態素解析し、単語をリストとして抽出する
            node = tagger.parseToNode(text)  # 最初のノードを取得
            words = []  # 単語を格納するリスト
            word_count = defaultdict(int)

            print('step2 pass')

            while node:  # ノードがNoneになるまでループ
              if node.surface.strip() != "":  # surfaceが空文字列でない場合（空白ノードを除外）
                pos = node.feature.split(",")[0]  # 品詞情報を取得
                if len(node.surface) > 1 and pos in ["名詞", "形容詞", "動詞"] :  # 1文字の単語と"名詞", "形容詞", "動詞"のどれか
                  words.append(node.surface)  # 単語をリストに追加
                  word_count[node.surface] += 1
              node = node.next  # 次のノードに移動

            # word_count辞書の最大値を確認
            max_word_count = max(word_count.values())

            # 最大値が30以上の場合、閾値を10に設定。それ以外の場合は、閾値を1に設定。
            threshold = 20 if max_word_count >= 30 else 1

            # 閾値に基づいて単語をフィルタリング
            filtered_words = [word for word in words if word_count[word] >= threshold]
             
            # 共起ネットワークを構築
            G = nx.Graph()

            print('step3 pass')

            for word1, word2 in combinations(set(filtered_words), 2):
              if G.has_edge(word1, word2):
                 G[word1][word2]['weight'] += 1
              else:
                 G.add_edge(word1, word2, weight=1)

            print('step4 pass')

            # 描画領域（FigureとAxes）を作成
            fig, ax = plt.subplots(figsize=(10, 10))

            print('step5 pass')

            # 共起ネットワークを描画
            draw_cooccurrence_network(G, ax=ax)

            print('step6 pass')

            st.pyplot(fig)

            print('step7 pass')

if uploaded_file is not None:
    raw_data = uploaded_file.getvalue()
    encoding = chardet.detect(raw_data)['encoding']
    text = raw_data.decode(encoding)

    # 処理の選択
    option = st.sidebar.selectbox("処理の種類を選択してください", ["ワードクラウド", "出現頻度表", "共起ネットワーク"])

    if option == "ワードクラウド":
        generate_wordcloud(text, font_path)
    elif option == "出現頻度表":
        generate_frequency_table(text)
    elif option == "共起ネットワーク":
        generate_cooccurrence_network(text, font_path)
else:
    st.write("ファイルがアップロードされていません。")
