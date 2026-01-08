"""
機器學習功能測試示例
ML Feature Testing Examples
"""
import requests
import json
import time
from typing import Dict

BASE_URL = "http://localhost:5000"


def print_section(title: str):
    """打印分隔線"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_result(title: str, result: Dict):
    """打印結果"""
    print(f"✓ {title}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def test_feature_generation():
    """測試特徵生成"""
    print_section("1. 測試特徵生成")
    
    response = requests.post(f"{BASE_URL}/api/ml/features/generate", json={
        'symbol': 'AAPL',
        'period': '2y'
    })
    
    result = response.json()
    if result['success']:
        stats = result['stats']
        print(f"✓ 成功生成 {stats['num_features']} 個特徵")
        print(f"  數據樣本數: {stats['num_samples']}")
        print(f"  前10個特徵: {stats['feature_names'][:10]}")
    else:
        print(f"✗ 失敗: {result.get('error')}")
    
    return result


def test_train_model(symbol: str = 'AAPL', model_type: str = 'random_forest'):
    """測試模型訓練"""
    print_section(f"2. 訓練 {model_type} 模型")
    
    print(f"正在訓練 {symbol} 的 {model_type} 模型...")
    print("這可能需要幾分鐘...\n")
    
    response = requests.post(f"{BASE_URL}/api/ml/train/advanced", json={
        'symbol': symbol,
        'period': '2y',
        'model_type': model_type,
        'tune_hyperparams': False,  # 快速測試，不調參
        'test_size': 0.2
    })
    
    result = response.json()
    if result['success']:
        print(f"✓ 模型訓練成功!")
        print(f"  模型ID: {result['model_id']}")
        print(f"  模型類型: {result['model_type']}")
        print(f"\n訓練集性能:")
        train = result['train_result']
        print(f"  準確率: {train['accuracy']:.2%}")
        print(f"  精確率: {train['precision']:.2%}")
        print(f"  召回率: {train['recall']:.2%}")
        print(f"  F1分數: {train['f1_score']:.2%}")
        print(f"\n測試集性能:")
        test = result['test_result']
        print(f"  準確率: {test['accuracy']:.2%}")
        print(f"  精確率: {test['precision']:.2%}")
        print(f"  召回率: {test['recall']:.2%}")
        print(f"  F1分數: {test['f1_score']:.2%}")
        if 'auc' in test and test['auc']:
            print(f"  AUC: {test['auc']:.2%}")
        
        print(f"\n前10個重要特徵:")
        importance = result['feature_importance']
        for i, (feature, value) in enumerate(list(importance.items())[:10], 1):
            print(f"  {i}. {feature}: {value:.4f}")
        
        return result['model_id']
    else:
        print(f"✗ 訓練失敗: {result.get('error')}")
        return None


def test_prediction(model_id: str, symbol: str = 'AAPL'):
    """測試模型預測"""
    print_section("3. 測試模型預測")
    
    response = requests.post(f"{BASE_URL}/api/ml/predict/advanced", json={
        'model_id': model_id,
        'symbol': symbol,
        'period': '3mo'
    })
    
    result = response.json()
    if result['success']:
        pred = result['latest_prediction']
        print(f"✓ 預測成功!")
        print(f"  股票: {symbol}")
        print(f"  信號: {pred['signal']}")
        print(f"  預測值: {pred['prediction']}")
        print(f"  概率: {pred['probability']}")
        print(f"  信心度: {pred['confidence']:.2%}")
        
        if pred['signal'] == 'BUY':
            emoji = "📈"
            suggestion = "建議買入"
        else:
            emoji = "📉"
            suggestion = "建議賣出"
        
        print(f"\n  {emoji} {suggestion} (信心度: {pred['confidence']:.2%})")
    else:
        print(f"✗ 預測失敗: {result.get('error')}")
    
    return result


def test_backtest(model_id: str, symbol: str = 'AAPL'):
    """測試ML策略回測"""
    print_section("4. 測試ML策略回測")
    
    print(f"正在回測 {symbol} 的ML策略...")
    print("這可能需要一些時間...\n")
    
    response = requests.post(f"{BASE_URL}/api/ml/backtest/ml_strategy", json={
        'model_id': model_id,
        'symbol': symbol,
        'period': '2y',
        'initial_capital': 1_000_000,
        'confidence_threshold': 0.6
    })
    
    result = response.json()
    if result['success']:
        bt = result['backtest_result']
        print(f"✓ 回測完成!")
        print(f"\n資金狀況:")
        print(f"  初始資金: ${bt['initial_capital']:,.0f}")
        print(f"  最終價值: ${bt['final_value']:,.0f}")
        print(f"  總收益: ${bt['total_return']:,.0f}")
        print(f"  收益率: {bt['total_return_pct']:.2f}%")
        
        print(f"\n交易統計:")
        print(f"  交易次數: {bt['num_trades']}")
        
        metrics = bt['metrics']
        print(f"  勝率: {metrics['win_rate']:.2f}%")
        print(f"  獲勝交易: {metrics['num_winning_trades']}")
        print(f"  虧損交易: {metrics['num_losing_trades']}")
        print(f"  平均獲利: ${metrics['avg_win']:,.0f}")
        print(f"  平均虧損: ${metrics['avg_loss']:,.0f}")
        print(f"  盈虧比: {metrics['profit_factor']:.2f}")
        
        print(f"\n風險指標:")
        print(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"  年化收益率: {metrics['annual_return_pct']:.2f}%")
        
        # 評分
        score = 0
        if bt['total_return_pct'] > 0:
            score += 1
        if metrics['win_rate'] > 50:
            score += 1
        if metrics['sharpe_ratio'] > 1:
            score += 1
        if metrics['max_drawdown'] < 20:
            score += 1
        
        print(f"\n策略評分: {'⭐' * score} ({score}/4)")
        
    else:
        print(f"✗ 回測失敗: {result.get('error')}")
    
    return result


def test_list_models():
    """測試列出所有模型"""
    print_section("5. 列出所有ML模型")
    
    response = requests.get(f"{BASE_URL}/api/ml/models")
    
    result = response.json()
    if result['success']:
        print(f"✓ 找到 {result['num_models']} 個模型:\n")
        
        for i, model in enumerate(result['models'], 1):
            print(f"{i}. {model['model_id']}")
            print(f"   類型: {model['model_type']}")
            print(f"   版本: {model['version']}")
            print(f"   創建時間: {model['created_at']}")
            if 'metadata' in model and model['metadata']:
                print(f"   元數據: {model['metadata']}")
            print()
    else:
        print(f"✗ 失敗: {result.get('error')}")
    
    return result


def test_compare_models(symbol: str = 'AAPL'):
    """測試比較多個模型"""
    print_section("6. 訓練並比較多個模型")
    
    model_types = ['random_forest', 'xgboost', 'lightgbm']
    model_ids = []
    
    # 訓練多個模型
    for model_type in model_types:
        print(f"訓練 {model_type} 模型...")
        response = requests.post(f"{BASE_URL}/api/ml/train/advanced", json={
            'symbol': symbol,
            'period': '2y',
            'model_type': model_type,
            'tune_hyperparams': False
        })
        
        result = response.json()
        if result['success']:
            model_ids.append(result['model_id'])
            print(f"  ✓ {result['model_id']}")
        else:
            print(f"  ✗ 失敗: {result.get('error')}")
        
        time.sleep(1)  # 避免請求過快
    
    if len(model_ids) < 2:
        print("\n需要至少2個模型才能比較")
        return
    
    # 比較模型
    print(f"\n比較 {len(model_ids)} 個模型...")
    response = requests.post(f"{BASE_URL}/api/ml/compare", json={
        'model_ids': model_ids,
        'symbol': symbol,
        'period': '1y'
    })
    
    result = response.json()
    if result['success']:
        comp = result['comparison']
        print(f"\n✓ 比較完成!")
        print(f"最佳模型: {comp['best_model']}\n")
        
        print("模型排名:")
        for i, model in enumerate(comp['comparisons'], 1):
            print(f"\n{i}. {model['model_type']}")
            print(f"   模型ID: {model['model_id']}")
            print(f"   準確率: {model['accuracy']:.2%}")
            print(f"   精確率: {model['precision']:.2%}")
            print(f"   召回率: {model['recall']:.2%}")
            print(f"   F1分數: {model['f1_score']:.2%}")
            if model['auc']:
                print(f"   AUC: {model['auc']:.2%}")
    else:
        print(f"✗ 比較失敗: {result.get('error')}")
    
    return result


def test_walk_forward():
    """測試滾動視窗分析"""
    print_section("7. 滾動視窗分析 (Walk-Forward)")
    
    print("正在進行滾動視窗分析...")
    print("這需要較長時間，因為要訓練多個模型...\n")
    
    response = requests.post(f"{BASE_URL}/api/ml/walk_forward", json={
        'symbol': 'AAPL',
        'period': '3y',
        'model_type': 'random_forest',
        'train_window': 252,
        'test_window': 63,
        'step_size': 63
    })
    
    result = response.json()
    if result['success']:
        wf = result['walk_forward_result']
        print(f"✓ 滾動分析完成!")
        print(f"\n總計分析了 {wf['num_periods']} 個時間段\n")
        
        print("各時間段表現:")
        for i, period in enumerate(wf['periods'], 1):
            print(f"\n時段 {i}: {period['test_start']} 至 {period['test_end']}")
            print(f"  收益率: {period['return_pct']:.2f}%")
            print(f"  交易次數: {period['num_trades']}")
        
        summary = wf['summary']
        print(f"\n總體統計:")
        print(f"  平均收益率: {summary['avg_return_pct']:.2f}%")
        print(f"  收益率標準差: {summary['std_return_pct']:.2f}%")
        print(f"  獲勝時段比例: {summary['win_rate']:.2f}%")
    else:
        print(f"✗ 分析失敗: {result.get('error')}")
    
    return result


def main():
    """主函數"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "機器學習功能測試" + " " * 10 + "║")
    print("║" + " " * 10 + "ML Feature Testing" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # 測試1: 特徵生成
        test_feature_generation()
        
        # 測試2: 訓練模型
        model_id = test_train_model('AAPL', 'random_forest')
        
        if model_id:
            # 測試3: 預測
            test_prediction(model_id, 'AAPL')
            
            # 測試4: 回測
            test_backtest(model_id, 'AAPL')
        
        # 測試5: 列出模型
        test_list_models()
        
        # 測試6: 比較模型 (可選，需要較長時間)
        # test_compare_models('AAPL')
        
        # 測試7: 滾動視窗分析 (可選，需要很長時間)
        # test_walk_forward()
        
        print_section("測試完成")
        print("✓ 所有測試已完成!")
        print("\n如需運行完整測試，請取消註釋 test_compare_models 和 test_walk_forward")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ 無法連接到服務器")
        print("請確保 app.py 正在運行: python app.py")
    except Exception as e:
        print(f"\n✗ 發生錯誤: {e}")


if __name__ == "__main__":
    main()
