import logging


class GlobalParams:
    """학습을 위한 파라미터를 정의한다."""

    debug: bool = False
    save_train_graph: bool = False
    data_dim = 5  # 입력 데이터 갯수
    output_dim = 1  # 출력 데이터 갯수
    learning_rate = 0.0001
    rmse_max = 0.5
    iterations = [10, 2000]  # 최소, 최대 훈련 반복횟수
    batch_size = 1024
    remove_stock_days = 0
    # stock_training_period_years = 21  # 훈련 데이터의 최대 년수
    # stock_test_period_years = 2 # 테스트 데이터의 년수
    max_listing_period_years = 20  # 상장일 기준 년수
    train_percent = 90.0  # 훈련 데이터 퍼센트
    dropout_keep = 0.6  # dropout
    train_batch_type = 'shuffle'  # 'sequential', 'shuffle', 'random', None
    # test_split_count = 10  # 테스트 데이터를 자르는 개수
    invest_start_date = '2019.09.16'
    invest_end_date = '2020.03.15'
    mock_period_months = 6
    invest_money = 10000000  # 각 주식에 모의투자할 금액
    index_money = None
    fee_percent = 0.015  # 투자시 발생하는 수수료
    tax_percent = 0.3  # 매도시 발생하는 세금
    # invest_min_percent = 0.315  # 투자를 하는 최소 간격 퍼센트
    invest_min_percent = 1.2  # 투자를 하는 최소 간격 퍼센트
    trade_min_percent = 3.0  # 투자 금액의 최소 간격 퍼센트
    invest_max_percent = 140  # 매매를 실행하는 예측율의 비율
    rule_trade_percent = 5.0  # 룰 베이스에서 매매를 실행하는 비율
    # kor_font_path = 'C:\\Windows\\Fonts\\korean.h2gtrm.ttf'
    kor_font_path = 'C:\\Windows\\Fonts\\HYGTRE.TTF'
    remove_session_file = False
    is_all_corps_model = False
    result_type = 'default'
    invest_type = 'default'
    invest_line_trading: bool = True
    check_stock_data = True
    check_kos_data = True
    result_file_name = None
    rmse_max_recommend = 0.03
    ensemble_type = 'rmse_square_ratio'  # rmse_ratio, rmse_square_ratio, rmse_best, avg, linear_regression

    exclude_corps = []
    # exclude_corps = [{'name': 'NAVER', 'code': '035420', 'month': '2018.10'},
    #                  {'name': '쌍용양회공업', 'code': '003410', 'month': '2018.07'},
    #                  {'name': '쌍용양회우', 'code': '003411', 'month': '2018.07'},
    #                  {'name': '코스모신소재', 'code': '005070', 'month': '2018.05'},
    #                  {'name': '휠라코리아', 'code': '081660', 'month': '2018.05'},
    #                  {'name': '까뮤이앤씨', 'code': '013700', 'month': '2018.05'},
    #                  {'name': '만도', 'code': '204320', 'month': '2018.05'},
    #                  {'name': 'KISCO홀딩스', 'code': '001940', 'month': '2018.05'},
    #                  {'name': '삼성전자', 'code': '005930', 'month': '2018.05'},
    #                  {'name': '삼성전자우', 'code': '005931', 'month': '2018.05'},
    #                  {'name': '한국프랜지', 'code': '010100', 'month': '2018.05'},
    #                  {'name': '한국철강', 'code': '104700', 'month': '2018.05'},
    #                  {'name': '보령제약', 'code': '003850', 'month': '2018.04'},
    #                  {'name': '한익스프레스', 'code': '014130', 'month': '2018.04'},
    #                  {'name': 'JW생명과학', 'code': '234080', 'month': '2018.04'},
    #                  {'name': '성지건설', 'code': '005980', 'month': '2018.01'},
    #                  {'name': '대한방직', 'code': '001070', 'month': '2018.01'}]

    def __init__(self, model_type='EACH', train_model='rnn'):
        self.model_type = model_type
        self.session_file_name = model_type
        self.train_model = train_model
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        if self.result_file_name is None:
            self.result_file_name = model_type.lower() + '_' + train_model + '_result'

        if train_model == 'rnn':
            self.loss_up_count = 20  # early stopping
            self.seq_length = 5  # 시퀀스 갯수
            self.hidden_dims = [128, 96, 64, 32, 16]
        elif train_model == 'cnn':
            self.loss_up_count = 10  # early stopping
            self.seq_length = 248
            self.kernel_size = 3
            self.pool_size = 2
            self.cnn_filters = [[16, 16], [32, 32], [48, 48], [64, 64], [128, 128]]
        elif train_model == 'mlp':
            self.loss_up_count = 10  # early stopping
            self.seq_length = 32  # 시퀀스 갯수
            self.hidden_dims = [128, 96, 64]

        if model_type == 'EACH':
            self.is_all_corps_model = False

        elif model_type == 'ALL_CORPS':
            self.is_all_corps_model = True

        # elif model_type == 'TWINS':
        #     self.is_all_corps_model = False

        elif model_type == 'FORCAST':
            self.is_all_corps_model = False
            # self.session_file_name = 'ALL_CORPS'
            #self.remove_session_file = True
            self.result_type = 'forcast'
            self.invest_start_date = None
            self.invest_end_date = None
            self.forcast_date = None
            self.check_stock_data = True
            self.remove_stock_days = 0

    def __str__(self):
        return "train_model:{}, model_type:{} seq_length:{}, hidden_dims:{}, cnn_filters:{}, kernel_size:{}, pool_size:{}".format(
            self.train_model, self.model_type, self.seq_length, self.hidden_dims, self.cnn_filters, self.kernel_size,
            self.pool_size)

