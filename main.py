import numpy as np
import torch

from cshogi import Board, BLACK, NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN, REPETITION_SUPERIOR, move_to_usi
from features import FEATURES_NUM, make_input_features, make_move_label
from uct_node import NodeTree
from policy_value_resnet import PolicyValueNetwork
from base_player import BasePlayer

from statistics import mean
import concurrent.futures
import time
import math


# デフォルトGPU ID
DEFAULT_GPU_ID = 0
# デフォルトバッチサイズ
DEFAULT_BATCH_SIZE = 32
# デフォルト投了閾値
DEFAULT_RESIGN_THRESHOLD = 0.01
# デフォルトPUCTの定数
DEFAULT_C_PUCT = 1.0
# デフォルト温度パラメータ
DEFAULT_TEMPERATURE = 1.0
# デフォルト持ち時間マージン(ms)
DEFAULT_TIME_MARGIN = 1000
# デフォルト秒読みマージン(ms)
DEFAULT_BYOYOMI_MARGIN = 100
# デフォルトPV表示間隔(ms)
DEFAULT_PV_INTERVAL = 500
# デフォルトプレイアウト数
DEFAULT_CONST_PLAYOUT = 1000
# 勝ちを表す定数（数値に意味はない）
VALUE_WIN = 10000
# 負けを表す定数（数値に意味はない）
VALUE_LOSE = -10000
# 引き分けを表す定数（数値に意味はない）
VALUE_DRAW = 20000
# キューに追加されたときの戻り値（数値に意味はない）
QUEUING = -1
# 探索を破棄するときの戻り値（数値に意味はない）
DISCARDED = -2
# Virtual Loss
VIRTUAL_LOSS = 1

# 各AIが選ばれた回数
ai_select_count = [0,0,0,0,0,0]
# 相手の勝率を格納するリストを追加
player_value_list = []
player_value_list2 = []
player_value_list3 = []
player_value_list4 = []
player_value_list5 = []
player_value_list6 = []
#評価値の格納
cp_collect = [0,0,0,0,0,0]


# 温度パラメータを適用した確率分布を取得
def softmax_temperature_with_normalize(logits, temperature):
    # 温度パラメータを適用
    logits /= temperature

    # 確率を計算(オーバーフローを防止するため最大値で引く)
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)

    # 合計が1になるように正規化
    sum_probabilities = sum(probabilities)
    probabilities /= sum_probabilities

    return probabilities

# ノード更新
def update_result(current_node, next_index, result):
    current_node.sum_value += result
    current_node.move_count += 1 - VIRTUAL_LOSS
    current_node.child_sum_value[next_index] += result
    current_node.child_move_count[next_index] += 1 - VIRTUAL_LOSS

# 評価待ちキューの要素
class EvalQueueElement:
    def set(self, node, color):
        self.node = node
        self.color = color

class MCTSPlayer(BasePlayer):
    # USIエンジンの名前
    name = 'multiple_player'        #変更
    # デフォルトチェックポイント
    DEFAULT_MODELFILE = "checkpoints/GCT_recent.pth"  #変更

    def __init__(self):
        super().__init__()
        # チェックポイントのパス
        self.modelfile = self.DEFAULT_MODELFILE

        # モデル
        #変更
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.model4 = None
        self.model5 = None
        self.model6 = None

        # 入力特徴量
        #変更
        self.features1 = None
        self.features2 = None
        self.features3 = None
        self.features4 = None
        self.features5 = None
        self.features6 = None

        # 評価待ちキュー
        #変更
        self.eval_queue1 = None
        self.eval_queue2 = None
        self.eval_queue3 = None
        self.eval_queue4 = None
        self.eval_queue5 = None
        self.eval_queue6 = None

        # バッチインデックス
        #変更
        self.current_batch_index1 = 0
        self.current_batch_index2 = 0
        self.current_batch_index3 = 0
        self.current_batch_index4 = 0
        self.current_batch_index5 = 0
        self.current_batch_index6 = 0

        # ルート局面
        self.root_board = Board()

        # ゲーム木
        #変更
        self.tree1 = NodeTree()
        self.tree2 = NodeTree()
        self.tree3 = NodeTree()
        self.tree4 = NodeTree()
        self.tree5 = NodeTree()
        self.tree6 = NodeTree()

        # プレイアウト回数
        #変更
        self.playout_count1 = 0
        self.playout_count2 = 0
        self.playout_count3 = 0
        self.playout_count4 = 0
        self.playout_count5 = 0
        self.playout_count6 = 0

        # 中断するプレイアウト回数
        self.halt = None

        # GPU ID
        self.gpu_id = DEFAULT_GPU_ID
        # デバイス
        self.device = None
        # バッチサイズ
        self.batch_size = DEFAULT_BATCH_SIZE
        # 投了する勝率の閾値
        self.resign_threshold = DEFAULT_RESIGN_THRESHOLD
        # PUCTの定数
        self.c_puct = DEFAULT_C_PUCT

        # 温度パラメータ
        # 変更
        self.temperature = DEFAULT_TEMPERATURE
        self.temperature2 = DEFAULT_TEMPERATURE
        self.temperature3 = DEFAULT_TEMPERATURE
        self.temperature4 = DEFAULT_TEMPERATURE
        self.temperature5 = DEFAULT_TEMPERATURE
        self.temperature6 = DEFAULT_TEMPERATURE


        # 持ち時間マージン(ms)
        self.time_margin = DEFAULT_TIME_MARGIN
        # 秒読みマージン(ms)
        self.byoyomi_margin = DEFAULT_BYOYOMI_MARGIN
        # PV表示間隔
        self.pv_interval = DEFAULT_PV_INTERVAL

        self.debug = False
        # 評価値
        self.cp = 0
        # 変更
        self.count = 0
        self.player_select = 0
        self.ai1_select = None
        self.ai2_select = None
        self.ai3_select = None
        self.ai4_select = None
        self.ai5_select = None
        self.ai6_select = None

        

###############
#USIプロトコル#
###############
    def usi(self):
        print('id name ' + self.name)
        print('option name USI_Ponder type check default false')
        print('option name modelfile type string default ' + self.DEFAULT_MODELFILE)
        print('option name gpu_id type spin default ' + str(DEFAULT_GPU_ID) + ' min -1 max 7')
        print('option name batchsize type spin default ' + str(DEFAULT_BATCH_SIZE) + ' min 1 max 256')
        print('option name resign_threshold type spin default ' + str(int(DEFAULT_RESIGN_THRESHOLD * 100)) + ' min 0 max 100')
        print('option name c_puct type spin default ' + str(int(DEFAULT_C_PUCT * 100)) + ' min 10 max 1000')
        print('option name temperature type spin default ' + str(int(DEFAULT_TEMPERATURE * 100)) + ' min 10 max 1000')
        print('option name time_margin type spin default ' + str(DEFAULT_TIME_MARGIN) + ' min 0 max 1000')
        print('option name byoyomi_margin type spin default ' + str(DEFAULT_BYOYOMI_MARGIN) + ' min 0 max 1000')
        print('option name pv_interval type spin default ' + str(DEFAULT_PV_INTERVAL) + ' min 0 max 10000')
        print('option name debug type check default false')

    def setoption(self, args):
        if args[1] == 'modelfile':
            self.modelfile = args[3]
        elif args[1] == 'gpu_id':
            self.gpu_id = int(args[3])
        elif args[1] == 'batchsize':
            self.batch_size = int(args[3])
        elif args[1] == 'resign_threshold':
            self.resign_threshold = int(args[3]) / 100
        elif args[1] == 'c_puct':
            self.c_puct = int(args[3]) / 100
        elif args[1] == 'temperature':
            #変更
            self.temperature = int(args[3]) / 100
            self.temperature2 = int(args[3]) / 100
            self.temperature3 = int(args[3]) / 100
            self.temperature4 = int(args[3]) / 100
            self.temperature5 = int(args[3]) / 100
            self.temperature6 = int(args[3]) / 100
        elif args[1] == 'time_margin':
            self.time_margin = int(args[3])
        elif args[1] == 'byoyomi_margin':
            self.byoyomi_margin = int(args[3])
        elif args[1] == 'pv_interval':
            self.pv_interval = int(args[3])
        elif args[1] == 'debug':
            self.debug = args[3] == 'true'

    # モデルのロード
    # 変更
    def load_model(self):
        self.model1 = PolicyValueNetwork()
        self.model1.to(self.device)
        checkpoint1 = torch.load(self.modelfile, map_location=self.device)
        self.model1.load_state_dict(checkpoint1['model'])
        # モデルを評価モードにする
        self.model1.eval()
        
        self.model2 = PolicyValueNetwork()
        self.model2.to(self.device)
        checkpoint2 = torch.load('checkpoints/floodgate0000.pth', map_location=self.device)
        #checkpoint2 = torch.load(self.modelfile, map_location=self.device)
        self.model2.load_state_dict(checkpoint2['model'])
        # モデルを評価モードにする
        self.model2.eval()

        self.model3 = PolicyValueNetwork()
        self.model3.to(self.device)
        checkpoint3 = torch.load('checkpoints/floodgate1000.pth', map_location=self.device)
        #checkpoint3 = torch.load(self.modelfile, map_location=self.device)
        self.model3.load_state_dict(checkpoint3['model'])
        # モデルを評価モードにする
        self.model3.eval()

        self.model4 = PolicyValueNetwork()
        self.model4.to(self.device)
        checkpoint4 = torch.load('checkpoints/floodgate2000.pth', map_location=self.device)
        #checkpoint4 = torch.load(self.modelfile, map_location=self.device)
        self.model4.load_state_dict(checkpoint4['model'])
        # モデルを評価モードにする
        self.model4.eval()

        self.model5 = PolicyValueNetwork()
        self.model5.to(self.device)
        checkpoint5 = torch.load('checkpoints/floodgate3000.pth', map_location=self.device)
        #checkpoint5 = torch.load(self.modelfile, map_location=self.device)
        self.model5.load_state_dict(checkpoint5['model'])
        # モデルを評価モードにする
        self.model5.eval()

        self.model6 = PolicyValueNetwork()
        self.model6.to(self.device)
        #checkpoint6 = torch.load('checkpoints/floodgate4000.pth', map_location=self.device)
        checkpoint6 = torch.load(self.modelfile, map_location=self.device)
        self.model6.load_state_dict(checkpoint6['model'])
        # モデルを評価モードにする
        self.model6.eval()

    # 入力特徴量の初期化
    # 変更
    def init_features(self):
        self.features1 = torch.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=(self.gpu_id >= 0))
        self.features2 = torch.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=(self.gpu_id >= 0))
        self.features3 = torch.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=(self.gpu_id >= 0))
        self.features4 = torch.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=(self.gpu_id >= 0))
        self.features5 = torch.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=(self.gpu_id >= 0))
        self.features6 = torch.empty((self.batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=(self.gpu_id >= 0))

    def isready(self):
        # デバイス
        if self.gpu_id >= 0:
            self.device = torch.device(f'cuda:{self.gpu_id}')
        else:
            self.device = torch.device('cpu')

        # モデルをロード
        self.load_model()

        # 変数の初期化
        # 評価値
        self.cp = 0
        # 変更
        self.count = 0
        self.player_select = None
        self.ai1_select = None
        self.ai2_select = None
        self.ai3_select = None
        self.ai4_select = None
        self.ai5_select = None
        self.ai6_select = None

        # 局面初期化
        # 変更
        self.root_board.reset()
        self.tree1.reset_to_position(self.root_board.zobrist_hash(), [])
        self.tree2.reset_to_position(self.root_board.zobrist_hash(), [])
        self.tree3.reset_to_position(self.root_board.zobrist_hash(), [])
        self.tree4.reset_to_position(self.root_board.zobrist_hash(), [])
        self.tree5.reset_to_position(self.root_board.zobrist_hash(), [])
        self.tree6.reset_to_position(self.root_board.zobrist_hash(), [])

        # 入力特徴量と評価待ちキューを初期化
        # 変更
        self.init_features()
        self.eval_queue1 = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index1 = 0
        self.eval_queue2 = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index2 = 0
        self.eval_queue3 = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index3 = 0
        self.eval_queue4 = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index4 = 0
        self.eval_queue5 = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index5 = 0
        self.eval_queue6 = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index6 = 0

        # モデルをキャッシュして初回推論を速くする
        # 変更
        current_node1 = self.tree1.current_head
        current_node1.expand_node(self.root_board)
        for _ in range(self.batch_size):
            self.queue_node1(self.root_board, current_node1)
        self.eval_node1()

        current_node2 = self.tree2.current_head
        current_node2.expand_node(self.root_board)
        for _ in range(self.batch_size):
            self.queue_node2(self.root_board, current_node2)
        self.eval_node2()

        current_node3 = self.tree3.current_head
        current_node3.expand_node(self.root_board)
        for _ in range(self.batch_size):
            self.queue_node3(self.root_board, current_node3)
        self.eval_node3()

        current_node4 = self.tree4.current_head
        current_node4.expand_node(self.root_board)
        for _ in range(self.batch_size):
            self.queue_node4(self.root_board, current_node4)
        self.eval_node4()

        current_node5 = self.tree5.current_head
        current_node5.expand_node(self.root_board)
        for _ in range(self.batch_size):
            self.queue_node5(self.root_board, current_node5)
        self.eval_node5()

        current_node6 = self.tree6.current_head
        current_node6.expand_node(self.root_board)
        for _ in range(self.batch_size):
            self.queue_node6(self.root_board, current_node6)
        self.eval_node6()

    # 変更
    def position(self, sfen, usi_moves):
        if sfen == 'startpos':
            self.root_board.reset()
        elif sfen[:5] == 'sfen ':
            self.root_board.set_sfen(sfen[5:])
        
        moves = []
        for usi_move in usi_moves:
            move = self.root_board.push_usi(usi_move)
            moves.append(move)
        self.tree1.reset_to_position(self.root_board.zobrist_hash(), moves)
        self.tree2.reset_to_position(self.root_board.zobrist_hash(), moves)
        self.tree3.reset_to_position(self.root_board.zobrist_hash(), moves)
        self.tree4.reset_to_position(self.root_board.zobrist_hash(), moves)
        self.tree5.reset_to_position(self.root_board.zobrist_hash(), moves)
        self.tree6.reset_to_position(self.root_board.zobrist_hash(), moves)
        self.count += 1
        if self.count != 1:
            self.player_select = moves[-1]
        else:
            self.player_select = None
        
  
    def set_limits(self, btime=None, wtime=None, byoyomi=None, binc=None, winc=None, nodes=None, infinite=False, ponder=False):
        # 探索回数の閾値を設定
        if infinite or ponder:
            # infiniteもしくはponderの場合は，探索を打ち切らないため，32bit整数の最大値を設定する
            self.halt = 2**31-1
        elif nodes:
            # プレイアウト数固定
            self.halt = nodes
        else:
            self.remaining_time, inc = (btime, binc) if self.root_board.turn == BLACK else (wtime, winc)
            if self.remaining_time is None and byoyomi is None and inc is None:
                # 時間指定がない場合
                self.halt = DEFAULT_CONST_PLAYOUT
            else:
                self.minimum_time = 0
                self.remaining_time = int(self.remaining_time) if self.remaining_time else 0
                inc = int(inc) if inc else 0
                self.time_limit = 100000
                # 秒読みの場合
                if byoyomi:
                    byoyomi = int(byoyomi) - self.byoyomi_margin
                    self.minimum_time = byoyomi
                    # time_limitが秒読み以下の場合，秒読みに設定
                    if self.time_limit < byoyomi:
                        self.time_limit = byoyomi
                self.halt = None

    def go(self):
        # 探索開始時刻の記録
        # 変更
        self.begin_time1 = time.time()
        self.begin_time2 = time.time()
        self.begin_time3 = time.time()
        self.begin_time4 = time.time()
        self.begin_time5 = time.time()
        self.begin_time6 = time.time()
        
        # 投了チェック
        if self.root_board.is_game_over():
            return 'resign', None
        '''
        # 入玉宣言勝ちチェック
        if self.root_board.is_nyugyoku():
            return 'win', None
        '''
        
        current_node1 = self.tree1.current_head
        current_node2 = self.tree2.current_head
        current_node3 = self.tree3.current_head
        current_node4 = self.tree4.current_head
        current_node5 = self.tree5.current_head
        current_node6 = self.tree6.current_head
        '''
        # 詰みの場合
        if current_node1.value == VALUE_WIN:
            matemove = self.root_board.mate_move(3)
            if matemove != 0:
                print('info score mate 3 pv {}'.format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None
        '''
        if not self.root_board.is_check():
            matemove = self.root_board.mate_move_in_1ply()
            if matemove:
                print('info score mate 1 pv {}'.format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None

        # プレイアウト数をクリア
        self.playout_count1 = 0
        self.playout_count2 = 0
        self.playout_count3 = 0
        self.playout_count4 = 0
        self.playout_count5 = 0
        self.playout_count6 = 0
        
        
        # ルートノードが未展開の場合，展開する
        if current_node1.child_move is None:
            current_node1.expand_node(self.root_board)
        if current_node2.child_move is None:
            current_node2.expand_node(self.root_board)
        if current_node3.child_move is None:
            current_node3.expand_node(self.root_board)
        if current_node4.child_move is None:
            current_node4.expand_node(self.root_board)
        if current_node5.child_move is None:
            current_node5.expand_node(self.root_board)
        if current_node6.child_move is None:
            current_node6.expand_node(self.root_board)
        
        # ルートノードが未評価の場合，評価する
        if current_node1.policy is None:
            self.current_batch_index1 = 0
            self.queue_node1(self.root_board, current_node1)
            self.eval_node1()
       
        if current_node2.policy is None:
            self.current_batch_index2 = 0
            self.queue_node2(self.root_board, current_node2)
            self.eval_node2()
        
        if current_node3.policy is None:
            self.current_batch_index3 = 0
            self.queue_node3(self.root_board, current_node3)
            self.eval_node3()
        
        if current_node4.policy is None:
            self.current_batch_index4 = 0
            self.queue_node4(self.root_board, current_node4)
            self.eval_node4()
        
        if current_node5.policy is None:
            self.current_batch_index5 = 0
            self.queue_node5(self.root_board, current_node5)
            self.eval_node5()
        
        if current_node6.policy is None:
            self.current_batch_index6 = 0
            self.queue_node6(self.root_board, current_node6)
            self.eval_node6()
        
        # 探索
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            executor.submit(self.search1)
            executor.submit(self.search2)
            executor.submit(self.search3)
            executor.submit(self.search4)
            executor.submit(self.search5)
            executor.submit(self.search6)
        if cp_collect[0] == -30000:
            return 'resign', None
        bestmove1, bestvalue1, ponder_move1 = self.get_bestmove_and_print_pv1()
        bestmove2, bestvalue2, ponder_move2 = self.get_bestmove_and_print_pv2()
        bestmove3, bestvalue3, ponder_move3 = self.get_bestmove_and_print_pv3()
        bestmove4, bestvalue4, ponder_move4 = self.get_bestmove_and_print_pv4()
        bestmove5, bestvalue5, ponder_move5 = self.get_bestmove_and_print_pv5()
        bestmove6, bestvalue6, ponder_move6 = self.get_bestmove_and_print_pv6()
    
        '''file = open('yosoku.txt','a')
        file.write(str(move_to_usi(ponder_move2))) if ponder_move2 else file.write(str('0000'))
        file.write(',')
        file.write(str(move_to_usi(ponder_move3))) if ponder_move3 else file.write(str('0000'))
        file.write(',')
        file.write(str(move_to_usi(ponder_move4))) if ponder_move4 else file.write(str('0000'))
        file.write(',')
        file.write(str(move_to_usi(ponder_move5))) if ponder_move5 else file.write(str('0000'))
        file.write(',')
        file.write(str(move_to_usi(ponder_move6))) if ponder_move6 else file.write(str('0000'))
        file.write('\n')
        file.close()'''
        
        
        if self.count < 2:
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            return move_to_usi(bestmove3), move_to_usi(ponder_move3) if ponder_move3 else None
        
        # 次の手と比較
        if self.player_select == self.ai2_select:
            print('AI_2')
            ai_select_count[1] += 1
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            return move_to_usi(bestmove2), move_to_usi(ponder_move2) if ponder_move2 else None
        elif self.player_select == self.ai3_select:
            print('AI_3')
            ai_select_count[0] += 1
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            return move_to_usi(bestmove3), move_to_usi(ponder_move3) if ponder_move3 else None
        elif self.player_select == self.ai4_select:
            print('AI_4')
            ai_select_count[3] += 1
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            return move_to_usi(bestmove4), move_to_usi(ponder_move4) if ponder_move4 else None
        elif self.player_select == self.ai5_select:
            print('AI_5')
            ai_select_count[4] += 1
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            return move_to_usi(bestmove5), move_to_usi(ponder_move5) if ponder_move5 else None
        elif self.player_select == self.ai6_select:
            print('AI_6')
            ai_select_count[5] += 1
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            return move_to_usi(bestmove6), move_to_usi(ponder_move6) if ponder_move6 else None
        elif sum(ai_select_count) == 0:
            print('AI_3')
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            return move_to_usi(bestmove3), move_to_usi(ponder_move3) if ponder_move3 else None
        elif self.cp >= 800 and self.count <= 60:
            print('AI_1')
            ai_select_count[2] += 1
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            return move_to_usi(bestmove1), move_to_usi(ponder_move1) if ponder_move1 else None
        else:
            self.ai1_select = ponder_move1
            self.ai2_select = ponder_move2
            self.ai3_select = ponder_move3
            self.ai4_select = ponder_move4
            self.ai5_select = ponder_move5
            self.ai6_select = ponder_move6
            max_index = ai_select_count.index(max(ai_select_count))
            if max_index == 1:
                print('AI_2')
                #ai_select_count[1] += 1
                return move_to_usi(bestmove2), move_to_usi(ponder_move2) if ponder_move2 else None
            elif max_index == 0:
                print('AI_3')
                #ai_select_count[0] += 1
                return move_to_usi(bestmove3), move_to_usi(ponder_move3) if ponder_move3 else None
            elif max_index == 3:
                print('AI_4')
                #ai_select_count[3] += 1
                return move_to_usi(bestmove4), move_to_usi(ponder_move4) if ponder_move4 else None
            elif max_index == 4:
                print('AI_5')
                #ai_select_count[4] += 1
                return move_to_usi(bestmove5), move_to_usi(ponder_move5) if ponder_move5 else None
            elif max_index == 5:
                print('AI_6')
                #ai_select_count[5] += 1
                return move_to_usi(bestmove6), move_to_usi(ponder_move6) if ponder_move6 else None
            else:
                print('AI_3')
                #ai_select_count[0] += 1
                return move_to_usi(bestmove3), move_to_usi(ponder_move3) if ponder_move3 else None

    def stop(self):
        # すぐに中断する
        self.halt = 0
        self.count = 0
        '''file = open('kekka.txt','a')    
        file.writelines(str(ai_select_count))
        file.write('\n')
        file.close()'''
        

    def ponderhit(self, last_limits):
        # 探索開始時刻の記録
        self.begin_time1 = time.time()
        self.last_pv_print_time1 = 0
        self.begin_time2 = time.time()
        self.last_pv_print_time2 = 0
        self.begin_time3 = time.time()
        self.last_pv_print_time3 = 0
        self.begin_time4 = time.time()
        self.last_pv_print_time4 = 0
        self.begin_time5 = time.time()
        self.last_pv_print_time5 = 0
        self.begin_time6 = time.time()
        self.last_pv_print_time6 = 0

        # プレイアウト数をクリア
        self.playout_count1 = 0
        self.playout_count2 = 0
        self.playout_count3 = 0
        self.playout_count4 = 0
        self.playout_count5 = 0
        self.playout_count6 = 0

        # 探索回数の閾値を設定
        self.set_limits(**last_limits)

    def quit(self):
        self.stop()

###############
#USIプロトコル#
###############

    def search1(self):
        self.last_pv_print_time1 = 0
        

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える，または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index1 = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                # 盤面のコピー
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search1(board, self.tree1.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.playout_count1 += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した，もしくはバックアップ済みのため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()
            
            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node1()
            
            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS
            
            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result
              
            # 探索を打ち切るか確認
            if self.check_interruption1():
                return
            
            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time1) * 1000)
                if elapsed_time > self.last_pv_print_time1 + self.pv_interval:
                    self.last_pv_print_time1 = elapsed_time
                    self.print_pv1()     #メソッドの変更
            

    # UCT探索
    def uct_search1(self, board, current_node, trajectories):
        # 子ノードのリストが初期化されていない場合，初期化する
        if not current_node.child_node:
            current_node.child_node = [None for _ in range(len(current_node.child_move))]
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.push(current_node.child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))

        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの作成
            child_node = current_node.create_child_node(next_index)

            # 千日手チェック
            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    # 千日手
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    # 連続王手の千日手で勝ち，もしくは優越局面の場合
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:   # draw == REPETITION_LOSE or draw == REPETITION_INFERIOR
                    # 連続王手の千日手で負け，もしくは劣等局面の場合
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                # 入玉宣言と3手詰めチェック
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    # 候補手を展開する
                    child_node.expand_node(board)
                    # 候補手がない場合
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        # ノードを評価待ちキューに追加
                        self.queue_node1(board, child_node)
                        return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search1(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return 1.0 - result
    
    # UCB値が最大の手を求める
    def select_max_ucb_child(self, node):
        q = np.divide(node.child_sum_value, node.child_move_count,
            out=np.zeros(len(node.child_move), np.float32),
            where=node.child_move_count != 0)
        if node.move_count == 0:
            u = 1.0
        else:
            u = np.sqrt(np.float32(node.move_count)) / (1 + node.child_move_count)
        ucb = q + self.c_puct * u * node.policy

        return np.argmax(ucb)

    #pvの表示のみ行うメソッド
    def print_pv1(self):
        finish_time = time.time() - self.begin_time1

        current_node = self.tree1.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        
        cp_collect[0] = cp
        
        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count1 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True) 

    def get_bestmove_and_print_pv1(self):
        finish_time = time.time() - self.begin_time1

        current_node = self.tree1.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        self.cp = abs(cp - int(-math.log(1.0 / (1.0 - bestvalue) - 1.0) * 600))
        cp_collect[0] = cp
        
        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count1 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)
        

        return bestmove, bestvalue, ponder_move
    
    # 探索を打ち切るか確認
    def check_interruption1(self):
        # プレイアウト回数が閾値を超えている
        if self.halt is not None:
            return self.playout_count1 >= self.halt
        '''
        # 候補手が1つの場合，中断する
        current_node = self.tree1.current_head
        if len(current_node.child_move) == 1:
            return True
        '''
        # 消費時間
        spend_time = int((time.time() - self.begin_time1) * 1000)

        # 消費時間が短すぎる場合，もしくは秒読みの場合は打ち切らない
        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False

        return True

    # 入力特徴量の作成
    def make_input_features1(self, board):
        make_input_features(board, self.features1.numpy()[self.current_batch_index1])

    # ノードをキューに追加
    def queue_node1(self, board, node):
        # 入力特徴量を作成
        self.make_input_features1(board)

        # ノードをキューに追加
        self.eval_queue1[self.current_batch_index1].set(node, board.turn)
        self.current_batch_index1 += 1
    
    # 推論
    def infer1(self):
        with torch.no_grad():
            x = self.features1[0:self.current_batch_index1].to(self.device)
            policy_logits, value_logits = self.model1(x)
            return policy_logits.cpu().numpy(), torch.sigmoid(value_logits).cpu().numpy()

    # 着手を表すラベル作成
    def make_move_label1(self, move, color):
        return make_move_label(move, color)
    
    # 局面の評価
    def eval_node1(self):
        # 推論
        policy_logits, values = self.infer1()

        for i, (policy_logit, value) in enumerate(zip(policy_logits, values)):
            current_node = self.eval_queue1[i].node
            color = self.eval_queue1[i].color

            # 合法手一覧
            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                move_label = self.make_move_label1(move, color)
                legal_move_probabilities[j] = policy_logit[move_label]

            # Boltzmann分布
            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, self.temperature)

            # ノードの値を更新
            current_node.policy = probabilities
            current_node.value = float(value)

##############
#以下繰り返し#
##############
#####
#AI2#
#####
    
    def search2(self):
        self.last_pv_print_time2 = 0

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える，または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index2 = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                # 盤面のコピー
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search2(board, self.tree2.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.playout_count2 += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した，もしくはバックアップ済みのため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node2()

            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result
            # 探索を打ち切るか確認
            if self.check_interruption2():
                return 
            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time2) * 1000)
                if elapsed_time > self.last_pv_print_time2 + self.pv_interval:
                    self.last_pv_print_time2 = elapsed_time
                    self.print_pv2()     #メソッドの変更

    def uct_search2(self, board, current_node, trajectories):
        # 子ノードのリストが初期化されていない場合，初期化する
        if not current_node.child_node:
            current_node.child_node = [None for _ in range(len(current_node.child_move))]
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.push(current_node.child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))
        
        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの作成
            child_node = current_node.create_child_node(next_index)

            # 千日手チェック
            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    # 千日手
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    # 連続王手の千日手で勝ち，もしくは優越局面の場合
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:   # draw == REPETITION_LOSE or draw == REPETITION_INFERIOR
                    # 連続王手の千日手で負け，もしくは劣等局面の場合
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                # 入玉宣言と3手詰めチェック
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    # 候補手を展開する
                    child_node.expand_node(board)
                    # 候補手がない場合
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        # ノードを評価待ちキューに追加
                        self.queue_node2(board, child_node)
                        return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search2(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return 1.0 - result
    
    def print_pv2(self):
        finish_time = time.time() - self.begin_time2

        current_node = self.tree2.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        cp_collect[1] = cp

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count2 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)
    
    def get_bestmove_and_print_pv2(self):
        # 探索にかかった時間を求める
        finish_time = time.time() - self.begin_time2

        #探索結果の勝率が相手の平均勝率に近く選択確率の高い手を選択
        current_node = self.tree2.current_head
        player_value = 1 - current_node.value
        player_value_list2.append(player_value)
        player_value_ave = mean(player_value_list2)

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None
        win_sort_idx = np.argsort(-win_rate)
        win_rate_diff = np.abs(player_value_ave - win_rate)
        selected_index = win_rate_diff.argsort()[0].tolist()
        for i in win_sort_idx:
            if win_rate_diff[i] <= 0.1:
                if current_node.policy[i] > current_node.policy[selected_index]:
                    selected_index = i

            elif np.isnan(win_rate[i]):
                break

        #勝率差が少ない手がない場合，すべての候補手から選択確率が最大の手を選択
        if win_rate_diff[selected_index] > 0.1:
            policy_list = np.zeros(len(current_node.child_move))
            for i in win_sort_idx:
                if np.isnan(win_rate[i]):
                    break
                else:
                    policy_list[i] = current_node.policy[i]
            selected_index = np.argmax(policy_list)


        # 選択した着手の勝率の算出
        bestvalue = win_rate[selected_index]

        bestmove = current_node.child_move[selected_index]
        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        
        cp_collect[1] = cp
        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count2 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)

        return bestmove, bestvalue, ponder_move
    
    def check_interruption2(self):
        # プレイアウト回数が閾値を超えている
        if self.halt is not None:
            return self.playout_count2 >= self.halt
        '''
        # 候補手が1つの場合，中断する
        current_node = self.tree2.current_head
        if len(current_node.child_move) == 1:
            return True
        '''
        # 消費時間
        spend_time = int((time.time() - self.begin_time2) * 1000)

        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False

        return True
    
    def make_input_features2(self, board):
        make_input_features(board, self.features2.numpy()[self.current_batch_index2])
    
    def queue_node2(self, board, node):
        # 入力特徴量を作成
        self.make_input_features2(board)

        # ノードをキューに追加
        self.eval_queue2[self.current_batch_index2].set(node, board.turn)
        self.current_batch_index2 += 1
    
    def infer2(self):
        with torch.no_grad():
            x = self.features2[0:self.current_batch_index2].to(self.device)
            policy_logits, value_logits = self.model2(x)
            return policy_logits.cpu().numpy(), torch.sigmoid(value_logits).cpu().numpy()
        
    def make_move_label2(self, move, color):
        return make_move_label(move, color)
    
    def eval_node2(self):
        # 推論
        policy_logits, values = self.infer2()

        for i, (policy_logit, value) in enumerate(zip(policy_logits, values)):
            current_node = self.eval_queue2[i].node
            color = self.eval_queue2[i].color

            # 合法手一覧
            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                move_label = self.make_move_label2(move, color)
                legal_move_probabilities[j] = policy_logit[move_label]

            # Boltzmann分布
            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, self.temperature2)

            # ノードの値を更新
            current_node.policy = probabilities
            current_node.value = float(value)

#####
#AI3#
#####

    def search3(self):
        self.last_pv_print_time3 = 0

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える，または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index3 = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                # 盤面のコピー
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search3(board, self.tree3.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.playout_count3 += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した，もしくはバックアップ済みのため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node3()

            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result
            # 探索を打ち切るか確認
            if self.check_interruption3():
                return 
            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time3) * 1000)
                if elapsed_time > self.last_pv_print_time3 + self.pv_interval:
                    self.last_pv_print_time3 = elapsed_time
                    self.print_pv3()     #メソッドの変更

    def uct_search3(self, board, current_node, trajectories):
        # 子ノードのリストが初期化されていない場合，初期化する
        if not current_node.child_node:
            current_node.child_node = [None for _ in range(len(current_node.child_move))]
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.push(current_node.child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))
        
        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの作成
            child_node = current_node.create_child_node(next_index)

            # 千日手チェック
            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    # 千日手
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    # 連続王手の千日手で勝ち，もしくは優越局面の場合
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:   # draw == REPETITION_LOSE or draw == REPETITION_INFERIOR
                    # 連続王手の千日手で負け，もしくは劣等局面の場合
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                # 入玉宣言と3手詰めチェック
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    # 候補手を展開する
                    child_node.expand_node(board)
                    # 候補手がない場合
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        # ノードを評価待ちキューに追加
                        self.queue_node3(board, child_node)
                        return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search3(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return 1.0 - result
    
    def print_pv3(self):
        finish_time = time.time() - self.begin_time3

        current_node = self.tree3.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        cp_collect[2] = cp
        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count3 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)
    
    def get_bestmove_and_print_pv3(self):
        finish_time = time.time() - self.begin_time3

        current_node = self.tree3.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        
        cp_collect[2] = cp

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count3 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)

        return bestmove, bestvalue, ponder_move
    
    def check_interruption3(self):
        # プレイアウト回数が閾値を超えている
        if self.halt is not None:
            return self.playout_count3 >= self.halt
        '''
        # 候補手が1つの場合，中断する
        current_node = self.tree3.current_head
        if len(current_node.child_move) == 1:
            return True
        '''
        # 消費時間
        spend_time = int((time.time() - self.begin_time3) * 1000)

        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False
        
        return True

    
    def make_input_features3(self, board):
        make_input_features(board, self.features3.numpy()[self.current_batch_index3])
    
    def queue_node3(self, board, node):
        # 入力特徴量を作成
        self.make_input_features3(board)

        # ノードをキューに追加
        self.eval_queue3[self.current_batch_index3].set(node, board.turn)
        self.current_batch_index3 += 1
    
    def infer3(self):
        with torch.no_grad():
            x = self.features3[0:self.current_batch_index3].to(self.device)
            policy_logits, value_logits = self.model3(x)
            return policy_logits.cpu().numpy(), torch.sigmoid(value_logits).cpu().numpy()
        
    def make_move_label3(self, move, color):
        return make_move_label(move, color)
    
    def eval_node3(self):
        # 推論
        policy_logits, values = self.infer3()

        for i, (policy_logit, value) in enumerate(zip(policy_logits, values)):
            current_node = self.eval_queue3[i].node
            color = self.eval_queue3[i].color

            # 合法手一覧
            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                move_label = self.make_move_label3(move, color)
                legal_move_probabilities[j] = policy_logit[move_label]

            # Boltzmann分布
            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, self.temperature3)

            # ノードの値を更新
            current_node.policy = probabilities
            current_node.value = float(value)

#####
#AI4#
#####

    def search4(self):
        self.last_pv_print_time4 = 0

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える，または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index4 = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                # 盤面のコピー
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search4(board, self.tree4.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.playout_count4 += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した，もしくはバックアップ済みのため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node4()

            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result
            # 探索を打ち切るか確認
            if self.check_interruption4():
                return 
            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time4) * 1000)
                if elapsed_time > self.last_pv_print_time4 + self.pv_interval:
                    self.last_pv_print_time4 = elapsed_time
                    self.print_pv4()     #メソッドの変更

    def uct_search4(self, board, current_node, trajectories):
        # 子ノードのリストが初期化されていない場合，初期化する
        if not current_node.child_node:
            current_node.child_node = [None for _ in range(len(current_node.child_move))]
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.push(current_node.child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))
        
        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの作成
            child_node = current_node.create_child_node(next_index)

            # 千日手チェック
            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    # 千日手
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    # 連続王手の千日手で勝ち，もしくは優越局面の場合
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:   # draw == REPETITION_LOSE or draw == REPETITION_INFERIOR
                    # 連続王手の千日手で負け，もしくは劣等局面の場合
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                # 入玉宣言と3手詰めチェック
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    # 候補手を展開する
                    child_node.expand_node(board)
                    # 候補手がない場合
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        # ノードを評価待ちキューに追加
                        self.queue_node4(board, child_node)
                        return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search4(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return 1.0 - result
    
    def print_pv4(self):
        finish_time = time.time() - self.begin_time4

        current_node = self.tree4.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        cp_collect[3] = cp

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count4 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)
    
    def get_bestmove_and_print_pv4(self):
        # 探索にかかった時間を求める
        finish_time = time.time() - self.begin_time4

        #探索結果の勝率が相手の平均勝率に近く選択確率の高い手を選択
        current_node = self.tree4.current_head
        player_value = 1 - current_node.value
        player_value_list4.append(player_value)
        player_value_ave = mean(player_value_list4)

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None
        win_sort_idx = np.argsort(-win_rate)
        win_rate_diff = np.abs(player_value_ave - win_rate)
        selected_index = win_rate_diff.argsort()[0].tolist()
        for i in win_sort_idx:
            if win_rate_diff[i] <= 0.1:
                if current_node.policy[i] > current_node.policy[selected_index]:
                    selected_index = i

            elif np.isnan(win_rate[i]):
                break

        #勝率差が少ない手がない場合，すべての候補手から選択確率が最大の手を選択
        if win_rate_diff[selected_index] > 0.1:
            policy_list = np.zeros(len(current_node.child_move))
            for i in win_sort_idx:
                if np.isnan(win_rate[i]):
                    break
                else:
                    policy_list[i] = current_node.policy[i]
            selected_index = np.argmax(policy_list)


        # 選択した着手の勝率の算出
        bestvalue = win_rate[selected_index]

        bestmove = current_node.child_move[selected_index]


        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        
        cp_collect[3] = cp

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count4 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)

        return bestmove, bestvalue, ponder_move
    
    def check_interruption4(self):
        # プレイアウト回数が閾値を超えている
        if self.halt is not None:
            return self.playout_count4 >= self.halt
        '''
        # 候補手が1つの場合，中断する
        current_node = self.tree4.current_head
        if len(current_node.child_move) == 1:
            return True
        '''
        # 消費時間
        spend_time = int((time.time() - self.begin_time4) * 1000)

        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False
        
        return True

    
    def make_input_features4(self, board):
        make_input_features(board, self.features4.numpy()[self.current_batch_index4])
    
    def queue_node4(self, board, node):
        # 入力特徴量を作成
        self.make_input_features4(board)

        # ノードをキューに追加
        self.eval_queue4[self.current_batch_index4].set(node, board.turn)
        self.current_batch_index4 += 1
    
    def infer4(self):
        with torch.no_grad():
            x = self.features4[0:self.current_batch_index4].to(self.device)
            policy_logits, value_logits = self.model4(x)
            return policy_logits.cpu().numpy(), torch.sigmoid(value_logits).cpu().numpy()
        
    def make_move_label4(self, move, color):
        return make_move_label(move, color)
    
    def eval_node4(self):
        # 推論
        policy_logits, values = self.infer4()

        for i, (policy_logit, value) in enumerate(zip(policy_logits, values)):
            current_node = self.eval_queue4[i].node
            color = self.eval_queue4[i].color

            # 合法手一覧
            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                move_label = self.make_move_label4(move, color)
                legal_move_probabilities[j] = policy_logit[move_label]

            # Boltzmann分布
            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, self.temperature4)

            # ノードの値を更新
            current_node.policy = probabilities
            current_node.value = float(value)

#####
#AI5#
#####

    def search5(self):
        self.last_pv_print_time5 = 0

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える，または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index5 = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                # 盤面のコピー
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search5(board, self.tree5.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.playout_count5 += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した，もしくはバックアップ済みのため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node5()

            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result
            # 探索を打ち切るか確認
            if self.check_interruption5():
                return 
            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time5) * 1000)
                if elapsed_time > self.last_pv_print_time5 + self.pv_interval:
                    self.last_pv_print_time5 = elapsed_time
                    self.print_pv5()     #メソッドの変更

    def uct_search5(self, board, current_node, trajectories):
        # 子ノードのリストが初期化されていない場合，初期化する
        if not current_node.child_node:
            current_node.child_node = [None for _ in range(len(current_node.child_move))]
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.push(current_node.child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))
        
        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの作成
            child_node = current_node.create_child_node(next_index)

            # 千日手チェック
            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    # 千日手
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    # 連続王手の千日手で勝ち，もしくは優越局面の場合
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:   # draw == REPETITION_LOSE or draw == REPETITION_INFERIOR
                    # 連続王手の千日手で負け，もしくは劣等局面の場合
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                # 入玉宣言と3手詰めチェック
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    # 候補手を展開する
                    child_node.expand_node(board)
                    # 候補手がない場合
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        # ノードを評価待ちキューに追加
                        self.queue_node5(board, child_node)
                        return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search5(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return 1.0 - result
    
    def print_pv5(self):
        finish_time = time.time() - self.begin_time5

        current_node = self.tree5.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        
        cp_collect[4] = cp

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count5 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)
    
    def get_bestmove_and_print_pv5(self):
        finish_time = time.time() - self.begin_time5

        current_node = self.tree5.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]


        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        
        cp_collect[4] = cp

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count5 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)

        return bestmove, bestvalue, ponder_move
    
    def check_interruption5(self):
        # プレイアウト回数が閾値を超えている
        if self.halt is not None:
            return self.playout_count5 >= self.halt
        '''
        # 候補手が1つの場合，中断する
        current_node = self.tree5.current_head
        if len(current_node.child_move) == 1:
            return True
        '''
        # 消費時間
        spend_time = int((time.time() - self.begin_time5) * 1000)

        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False
        
        return True

    
    def make_input_features5(self, board):
        make_input_features(board, self.features5.numpy()[self.current_batch_index5])
    
    def queue_node5(self, board, node):
        # 入力特徴量を作成
        self.make_input_features5(board)

        # ノードをキューに追加
        self.eval_queue5[self.current_batch_index5].set(node, board.turn)
        self.current_batch_index5 += 1
    
    def infer5(self):
        with torch.no_grad():
            x = self.features5[0:self.current_batch_index5].to(self.device)
            policy_logits, value_logits = self.model5(x)
            return policy_logits.cpu().numpy(), torch.sigmoid(value_logits).cpu().numpy()
        
    def make_move_label5(self, move, color):
        return make_move_label(move, color)
    
    def eval_node5(self):
        # 推論
        policy_logits, values = self.infer5()

        for i, (policy_logit, value) in enumerate(zip(policy_logits, values)):
            current_node = self.eval_queue5[i].node
            color = self.eval_queue5[i].color

            # 合法手一覧
            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                move_label = self.make_move_label5(move, color)
                legal_move_probabilities[j] = policy_logit[move_label]

            # Boltzmann分布
            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, self.temperature5)

            # ノードの値を更新
            current_node.policy = probabilities
            current_node.value = float(value)

#####
#AI6#
#####

    def search6(self):
        self.last_pv_print_time6 = 0

        # 探索経路のバッチ
        trajectories_batch = []
        trajectories_batch_discarded = []

        # 探索回数が閾値を超える，または探索が打ち切られたらループを抜ける
        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index6 = 0

            # バッチサイズの回数だけシミュレーションを行う
            for i in range(self.batch_size):
                # 盤面のコピー
                board = self.root_board.copy()

                # 探索
                trajectories_batch.append([])
                result = self.uct_search6(board, self.tree6.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.playout_count6 += 1
                else:
                    # 破棄した探索経路を保存
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    # 破棄が多い場合はすぐに評価する
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                # 評価中の葉ノードに達した，もしくはバックアップ済みのため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            # 評価
            if len(trajectories_batch) > 0:
                self.eval_node6()

            # 破棄した探索経路のVirtual Lossを戻す
            for trajectories in trajectories_batch_discarded:
                for i in range(len(trajectories)):
                    current_node, next_index = trajectories[i]
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            # バックアップ
            for trajectories in trajectories_batch:
                result = None
                for i in reversed(range(len(trajectories))):
                    current_node, next_index = trajectories[i]
                    if result is None:
                        # 葉ノード
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result
            # 探索を打ち切るか確認
            if self.check_interruption6():
                return 
            # PV表示
            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time6) * 1000)
                if elapsed_time > self.last_pv_print_time6 + self.pv_interval:
                    self.last_pv_print_time6 = elapsed_time
                    self.print_pv6()     #メソッドの変更

    def uct_search6(self, board, current_node, trajectories):
        # 子ノードのリストが初期化されていない場合，初期化する
        if not current_node.child_node:
            current_node.child_node = [None for _ in range(len(current_node.child_move))]
        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        board.push(current_node.child_move[next_index])

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS

        # 経路を記録
        trajectories.append((current_node, next_index))
        
        # ノードの展開の確認
        if current_node.child_node[next_index] is None:
            # ノードの作成
            child_node = current_node.create_child_node(next_index)

            # 千日手チェック
            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    # 千日手
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    # 連続王手の千日手で勝ち，もしくは優越局面の場合
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:   # draw == REPETITION_LOSE or draw == REPETITION_INFERIOR
                    # 連続王手の千日手で負け，もしくは劣等局面の場合
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                # 入玉宣言と3手詰めチェック
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    # 候補手を展開する
                    child_node.expand_node(board)
                    # 候補手がない場合
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        # ノードを評価待ちキューに追加
                        self.queue_node6(board, child_node)
                        return QUEUING
        else:
            # 評価待ちのため破棄する
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED

            # 詰みと千日手チェック
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                # 手番を入れ替えて1手深く読む
                result = self.uct_search6(board, next_node, trajectories)

        if result == QUEUING or result == DISCARDED:
            return result

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return 1.0 - result
    
    def print_pv6(self):
        finish_time = time.time() - self.begin_time6

        current_node = self.tree6.current_head
        player_value = 1 - current_node.value

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None

        selected_index = np.abs(win_rate - player_value).argsort()[0].tolist()
        bestvalue = current_node.child_sum_value[selected_index] / current_node.child_move_count[selected_index]
        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        
        cp_collect[5] = cp

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count6 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)
    
    def get_bestmove_and_print_pv6(self):
        # 探索にかかった時間を求める
        finish_time = time.time() - self.begin_time6

        #探索結果の勝率が相手の平均勝率に近く選択確率の高い手を選択
        current_node = self.tree6.current_head
        player_value = 1 - current_node.value
        player_value_list6.append(player_value)
        player_value_ave = mean(player_value_list6)

        win_rate = np.zeros(len(current_node.child_move))
        for i in range(len(current_node.child_move)):
            if current_node.child_move_count[i] > 0:
                win_rate[i] = current_node.child_sum_value[i] / current_node.child_move_count[i]
            else:
                win_rate[i] = None
        win_sort_idx = np.argsort(-win_rate)
        win_rate_diff = np.abs(player_value_ave - win_rate)
        selected_index = win_rate_diff.argsort()[0].tolist()
        for i in win_sort_idx:
            if win_rate_diff[i] <= 0.1:
                if current_node.policy[i] > current_node.policy[selected_index]:
                    selected_index = i

            elif np.isnan(win_rate[i]):
                break

        #勝率差が少ない手がない場合，すべての候補手から選択確率が最大の手を選択
        if win_rate_diff[selected_index] > 0.1:
            policy_list = np.zeros(len(current_node.child_move))
            for i in win_sort_idx:
                if np.isnan(win_rate[i]):
                    break
                else:
                    policy_list[i] = current_node.policy[i]
            selected_index = np.argmax(policy_list)


        # 選択した着手の勝率の算出
        bestvalue = win_rate[selected_index]

        bestmove = current_node.child_move[selected_index]

        # 勝率を評価値に変換
        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)
        
        cp_collect[5] = cp

        # PV
        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = np.argmax(pv_node.child_move_count)
            pv += ' ' + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(self.playout_count6 / finish_time) if finish_time > 0 else 0,
            int(finish_time * 1000),
            current_node.move_count,
            cp, pv), flush=True)

        return bestmove, bestvalue, ponder_move
    
    def check_interruption6(self):
        # プレイアウト回数が閾値を超えている
        if self.halt is not None:
            return self.playout_count6 >= self.halt
        '''
        # 候補手が1つの場合，中断する
        current_node = self.tree6.current_head
        if len(current_node.child_move) == 1:
            return True
        '''
        # 消費時間
        spend_time = int((time.time() - self.begin_time6) * 1000)

        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False
        
        return True

    
    def make_input_features6(self, board):
        make_input_features(board, self.features6.numpy()[self.current_batch_index6])
    
    def queue_node6(self, board, node):
        # 入力特徴量を作成
        self.make_input_features6(board)

        # ノードをキューに追加
        self.eval_queue6[self.current_batch_index6].set(node, board.turn)
        self.current_batch_index6 += 1
    
    def infer6(self):
        with torch.no_grad():
            x = self.features6[0:self.current_batch_index6].to(self.device)
            policy_logits, value_logits = self.model6(x)
            return policy_logits.cpu().numpy(), torch.sigmoid(value_logits).cpu().numpy()
        
    def make_move_label6(self, move, color):
        return make_move_label(move, color)
    
    def eval_node6(self):
        # 推論
        policy_logits, values = self.infer6()

        for i, (policy_logit, value) in enumerate(zip(policy_logits, values)):
            current_node = self.eval_queue6[i].node
            color = self.eval_queue6[i].color

            # 合法手一覧
            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
            for j in range(len(current_node.child_move)):
                move = current_node.child_move[j]
                move_label = self.make_move_label6(move, color)
                legal_move_probabilities[j] = policy_logit[move_label]

            # Boltzmann分布
            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, self.temperature6)

            # ノードの値を更新
            current_node.policy = probabilities
            current_node.value = float(value)
            
if __name__ == '__main__':
    player = MCTSPlayer()
    player.run()