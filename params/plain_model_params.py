

class PlainModelParams:
    debug = False
    data_dim = 5  # 입력 데이터 갯수
    output_dim = 1  # 출력 데이터 갯수
    learning_rate = 0.0001
    remove_stock_days = 2
    train_percent = 90
    invest_count = 160
    invest_money = 10000000  # 각 주식에 모의투자할 금액
    fee_percent = 0.015  # 투자시 발생하는 수수료
    tax_percent = 0.3  # 매도시 발생하는 세금
    invest_min_percent = 1.2  # 투자를 하는 최소 간격 퍼센트
    invest_max_percent = 140
    result_file_name = None
    check_stock_data = False
    kor_font_path = 'C:\\WINDOWS\\Fonts\\H2GTRM.TTF'
    train_model = 'plain'

    # decision_tree, random_forest, linear_regression
    def __init__(self, model_type='decision_tree'):
        self.model_type = model_type
        if model_type == 'decision_tree':
            pass
        elif model_type == 'decision_tree':
            pass
        elif model_type == 'linear_regression':
            pass
