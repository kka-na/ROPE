"""
Module 3: Event Analysis
- Event Analysis: Signal-based event classification and detection delay analysis
- Note: EKF-based uncertainty analysis has been moved to uncertainty_module.py
"""

import pandas as pd
import numpy as np
from typing import Dict


class EventAndUncertaintyAnalyzer:
    """이벤트 분석 클래스 (불확실성 분석은 UncertaintyAnalyzer로 분리됨)"""
    
    # Signal/Event 타입 정의
    SIGNAL_TYPES = {
        0: "none",
        1: "lane_change_left",
        2: "lane_change_right",
        4: "merge_accept",
        5: "merge_reject",
        7: "emergency"
    }
    
    def __init__(self):
        pass
    
    def analyze_events(self, df: pd.DataFrame) -> Dict:
        """신호/이벤트 분석 - 감지 지연 중심 (통신 지연)"""
        
        events = {
            "signal_distribution": {},
            "signal_detection_delay": {},
            "signal_detection_delay_stats": {},
            "message_delay_analysis": {}
        }
        
        # Signal 분포 (4개 신호 모두)
        for col_name, col_data in [("ego_signal", df['ego_signal']), 
                                    ("target_signal", df['target_signal']),
                                    ("v2v_signal", df['v2v_signal']),
                                    ("target_v2v_signal", df['target_v2v_signal'])]:
            # NaN 값 제외
            valid_signals = col_data[pd.notna(col_data)]
            if len(valid_signals) > 0:
                signal_counts = valid_signals.value_counts().to_dict()
                events["signal_distribution"][col_name] = {
                    self.SIGNAL_TYPES.get(int(k), f"unknown_{k}"): int(v) 
                    for k, v in signal_counts.items()
                }
        
        # ===== 신호 쌍 정의 (통신 지연 측정) =====
        # 1. ego_signal (ego 발생) → target_v2v_signal (target 수신) = ego→target 통신 지연
        # 2. target_signal (target 발생) → v2v_signal (ego 수신) = target→ego 통신 지연

        signal_pairs = [
            {
                "name": "ego→target_communication",
                "source_col": "ego_signal",
                "target_col": "target_v2v_signal",
                "source_ts": df['ts_ns'].values,
                "target_ts": df['target_v2v_ts'].values,
                "description": "Ego 신호 발생 → Target 수신 (통신 지연)"
            },
            {
                "name": "target→ego_communication",
                "source_col": "target_signal",
                "target_col": "v2v_signal",
                "source_ts": df['target_ts'].values,
                "target_ts": df['ts_ns'].values,
                "description": "Target 신호 발생 → Ego 수신 (통신 지연)"
            }
        ]
        
        # 각 신호 쌍에 대해 detection delay 계산
        for pair_info in signal_pairs:
            source_name = pair_info["source_col"]
            target_name = pair_info["target_col"]
            pair_key = pair_info["name"]
            source_ts = pair_info["source_ts"]
            target_ts = pair_info["target_ts"]
            
            detection_delays = []
            
            if source_name in df.columns and target_name in df.columns:
                source_sig = df[source_name].values
                target_sig = df[target_name].values
                
                # Source signal 변화 찾기 (이전 값이 0 또는 NaN, 현재 값이 non-zero)
                source_transitions = []
                for i in range(1, len(source_sig)):
                    # 현재 값이 유효하고 non-zero
                    # 이전 값이 0이거나 NaN
                    if pd.notna(source_sig[i]) and source_sig[i] != 0 and \
                       (source_sig[i-1] == 0 or pd.isna(source_sig[i-1])):
                        source_transitions.append({
                            "index": i,
                            "timestamp_ns": source_ts[i],
                            "from": 0,  # 이전 값을 0으로 통일
                            "to": int(source_sig[i]),
                            "time_s": (source_ts[i] - source_ts[0]) / 1e9
                        })
                
                # Target signal 변화 찾기 (이전 값이 0 또는 NaN, 현재 값이 non-zero)
                target_transitions = []
                for i in range(1, len(target_sig)):
                    # 현재 값이 유효하고 non-zero
                    # 이전 값이 0이거나 NaN
                    if pd.notna(target_sig[i]) and target_sig[i] != 0 and \
                       (target_sig[i-1] == 0 or pd.isna(target_sig[i-1])):
                        target_transitions.append({
                            "index": i,
                            "timestamp_ns": target_ts[i],
                            "from": 0,  # 이전 값을 0으로 통일
                            "to": int(target_sig[i]),
                            "time_s": (target_ts[i] - target_ts[0]) / 1e9
                        })
                
                # Source signal 변화마다 가장 가까운 Target signal 변화와의 지연 계산
                for source_trans in source_transitions:
                    matching_target = None
                    min_delay = float('inf')
                    
                    for target_trans in target_transitions:
                        # 신호와의 시간 차이 계산 (음수 가능, 절대값으로 최소값 찾기)
                        delay = (target_trans["timestamp_ns"] - source_trans["timestamp_ns"]) / 1e6  # ns → ms
                        if abs(delay) < abs(min_delay):
                            min_delay = abs(delay)
                            matching_target = target_trans
                    
                    if matching_target is not None:
                        detection_delays.append(min_delay)
                        
                        if pair_key not in events["signal_detection_delay"]:
                            events["signal_detection_delay"][pair_key] = []
                        
                        events["signal_detection_delay"][pair_key].append({
                            f"{source_name}_change": f"{self.SIGNAL_TYPES.get(source_trans['from'])} → {self.SIGNAL_TYPES.get(source_trans['to'])}",
                            f"{source_name}_time_s": float(source_trans["time_s"]),
                            f"{target_name}_change": f"{self.SIGNAL_TYPES.get(matching_target['from'])} → {self.SIGNAL_TYPES.get(matching_target['to'])}",
                            f"{target_name}_time_s": float(matching_target["time_s"]),
                            "communication_delay_ms": float(min_delay)
                        })
                
                # 통신 지연 통계
                if detection_delays:
                    detection_delays = np.array(detection_delays)
                    events["signal_detection_delay_stats"][pair_key] = {
                        "description": pair_info["description"],
                        "count": int(len(detection_delays)),
                        "mean_delay_ms": float(np.mean(detection_delays)),
                        "median_delay_ms": float(np.median(detection_delays)),
                        "std_delay_ms": float(np.std(detection_delays)),
                        "min_delay_ms": float(np.min(detection_delays)),
                        "max_delay_ms": float(np.max(detection_delays)),
                        "p25_delay_ms": float(np.percentile(detection_delays, 25)),
                        "p75_delay_ms": float(np.percentile(detection_delays, 75)),
                        "p95_delay_ms": float(np.percentile(detection_delays, 95))
                    }
                else:
                    events["signal_detection_delay_stats"][pair_key] = {
                        "description": pair_info["description"],
                        "count": 0,
                        "note": f"{source_name}과 {target_name} 신호 변화가 일치하지 않음 (NaN 또는 신호 변화 없음)"
                    }
        
        # Message delay analysis (기존)
        if 'dt_rx_tx_ms' in df.columns:
            dt_rx_tx = df['dt_rx_tx_ms'].values
            # NaN 제외
            valid_delay = dt_rx_tx[np.isfinite(dt_rx_tx)]
            if len(valid_delay) > 0:
                events["message_delay_analysis"] = {
                    "mean_delay_ms": float(np.mean(valid_delay)),
                    "median_delay_ms": float(np.median(valid_delay)),
                    "std_delay_ms": float(np.std(valid_delay)),
                    "min_delay_ms": float(np.min(valid_delay)),
                    "max_delay_ms": float(np.max(valid_delay)),
                    "p95_delay_ms": float(np.percentile(valid_delay, 95))
                }
        
        return events
    
    def analyze_position_velocity_comparison(self, df: pd.DataFrame) -> Dict:
        """V2V vs Target, Lidar vs Target 위치/속도 비교 분석"""
        
        results = {}
        
        # 1. V2V vs Target 비교 (Position, Velocity, Heading)
        if all(col in df.columns for col in ['v2v_enu_x', 'v2v_enu_y', 'target_enu_x', 'target_enu_y', 'v2v_velocity', 'target_velocity', 'v2v_h', 'target_h']):
            v2v_x = df['v2v_enu_x'].values
            v2v_y = df['v2v_enu_y'].values
            target_x = df['target_enu_x'].values
            target_y = df['target_enu_y'].values
            v2v_vel = df['v2v_velocity'].values
            target_vel = df['target_velocity'].values
            v2v_h = df['v2v_h'].values
            target_h = df['target_h'].values
            
            # NaN 제외
            valid_mask = np.isfinite(v2v_x) & np.isfinite(v2v_y) & np.isfinite(target_x) & np.isfinite(target_y)
            
            if valid_mask.any():
                pos_diff = np.sqrt((v2v_x[valid_mask] - target_x[valid_mask])**2 + 
                                   (v2v_y[valid_mask] - target_y[valid_mask])**2)
                
                vel_diff = np.abs(v2v_vel[valid_mask] - target_vel[valid_mask])
                
                # Heading 차이 (라디안)
                h_diff = np.deg2rad(v2v_h[valid_mask] - target_h[valid_mask])
                
                results["v2v_vs_target"] = {
                    "position_rmse_m": float(np.sqrt(np.mean(pos_diff**2))),
                    "position_mean_m": float(np.mean(pos_diff)),
                    "position_median_m": float(np.median(pos_diff)),
                    "position_std_m": float(np.std(pos_diff)),
                    "velocity_rmse_ms": float(np.sqrt(np.mean(vel_diff**2))),
                    "velocity_mean_ms": float(np.mean(vel_diff)),
                    "velocity_std_ms": float(np.std(vel_diff)),
                    "heading_rmse_deg": float(np.sqrt(np.mean(h_diff**2)) * 180 / np.pi),
                    "heading_mean_deg": float(np.mean(h_diff) * 180 / np.pi),
                    "heading_std_deg": float(np.std(h_diff) * 180 / np.pi)
                }
        
        # 2. Lidar vs Target 비교 (Position, Velocity, Heading)
        if all(col in df.columns for col in ['lidar_0_enu_x', 'lidar_0_enu_y', 'target_enu_x', 'target_enu_y', 'lidar_0_velocity', 'target_velocity', 'lidar_0_h', 'target_h']):
            lidar_x = df['lidar_0_enu_x'].values
            lidar_y = df['lidar_0_enu_y'].values
            target_x = df['target_enu_x'].values
            target_y = df['target_enu_y'].values
            lidar_vel = df['lidar_0_velocity'].values
            target_vel = df['target_velocity'].values
            lidar_h = df['lidar_0_h'].values
            target_h = df['target_h'].values
            
            # NaN 제외
            valid_mask = np.isfinite(lidar_x) & np.isfinite(lidar_y) & np.isfinite(target_x) & np.isfinite(target_y) & \
                         np.isfinite(lidar_vel) & np.isfinite(target_vel) & np.isfinite(lidar_h) & np.isfinite(target_h)
            
            if valid_mask.any():
                pos_diff = np.sqrt((lidar_x[valid_mask] - target_x[valid_mask])**2 + 
                                   (lidar_y[valid_mask] - target_y[valid_mask])**2)
                
                vel_diff = np.abs(lidar_vel[valid_mask] - target_vel[valid_mask])
                
                # Heading 차이 (라디안)
                h_diff = np.deg2rad(lidar_h[valid_mask] - target_h[valid_mask])
                
                results["lidar_vs_target"] = {
                    "valid_samples": int(np.sum(valid_mask)),
                    "position_rmse_m": float(np.sqrt(np.mean(pos_diff**2))),
                    "position_mean_m": float(np.mean(pos_diff)),
                    "position_median_m": float(np.median(pos_diff)),
                    "position_std_m": float(np.std(pos_diff)),
                    "position_min_m": float(np.min(pos_diff)),
                    "position_max_m": float(np.max(pos_diff)),
                    "velocity_rmse_ms": float(np.sqrt(np.mean(vel_diff**2))),
                    "velocity_mean_ms": float(np.mean(vel_diff)),
                    "velocity_std_ms": float(np.std(vel_diff)),
                    "heading_rmse_deg": float(np.sqrt(np.mean(h_diff**2)) * 180 / np.pi),
                    "heading_mean_deg": float(np.mean(h_diff) * 180 / np.pi),
                    "heading_std_deg": float(np.std(h_diff) * 180 / np.pi)
                }
        
        # 3. V2V vs Lidar Position/Velocity/Heading Sync
        if all(col in df.columns for col in ['v2v_enu_x', 'v2v_enu_y', 'lidar_0_enu_x', 'lidar_0_enu_y', 'v2v_velocity', 'lidar_0_velocity', 'v2v_h', 'lidar_0_h']):
            v2v_x = df['v2v_enu_x'].values
            v2v_y = df['v2v_enu_y'].values
            lidar_x = df['lidar_0_enu_x'].values
            lidar_y = df['lidar_0_enu_y'].values
            v2v_vel = df['v2v_velocity'].values
            lidar_vel = df['lidar_0_velocity'].values
            v2v_h = df['v2v_h'].values
            lidar_h = df['lidar_0_h'].values
            
            # NaN 제외
            valid_mask = np.isfinite(v2v_x) & np.isfinite(v2v_y) & np.isfinite(lidar_x) & np.isfinite(lidar_y) & \
                         np.isfinite(v2v_vel) & np.isfinite(lidar_vel) & np.isfinite(v2v_h) & np.isfinite(lidar_h)
            
            if valid_mask.any():
                pos_sync = np.sqrt((v2v_x[valid_mask] - lidar_x[valid_mask])**2 + 
                                   (v2v_y[valid_mask] - lidar_y[valid_mask])**2)
                
                vel_sync = np.abs(v2v_vel[valid_mask] - lidar_vel[valid_mask])
                
                # Heading 차이
                h_sync = np.deg2rad(v2v_h[valid_mask] - lidar_h[valid_mask])
                
                results["v2v_vs_lidar_position_sync"] = {
                    "valid_samples": int(np.sum(valid_mask)),
                    "position_sync_error_mean_m": float(np.mean(pos_sync)),
                    "position_sync_error_median_m": float(np.median(pos_sync)),
                    "position_sync_error_std_m": float(np.std(pos_sync)),
                    "position_sync_error_min_m": float(np.min(pos_sync)),
                    "position_sync_error_max_m": float(np.max(pos_sync)),
                    "velocity_sync_error_mean_ms": float(np.mean(vel_sync)),
                    "velocity_sync_error_median_ms": float(np.median(vel_sync)),
                    "velocity_sync_error_std_ms": float(np.std(vel_sync)),
                    "heading_sync_error_mean_deg": float(np.mean(h_sync) * 180 / np.pi),
                    "heading_sync_error_median_deg": float(np.median(h_sync) * 180 / np.pi),
                    "heading_sync_error_std_deg": float(np.std(h_sync) * 180 / np.pi)
                }
        
        return results