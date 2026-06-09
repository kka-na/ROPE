"""
V2VBridge: OBU 하드웨어 소켓(v2x/libs/socket_handler.py) 대체.
소프트웨어 채널 에뮬레이터로 Delay·PRR 주입.

WC 모드: ego/EgoShareInfo → 지연+손실 처리 → target/TargetShareInfo (와 반대)
WOC 모드: --woc 플래그 시 아무것도 forwarding 안 함
"""
import rospy, time, random, threading, yaml, os, argparse, math
from collections import deque
from ccavt.msg import ShareInfo
from std_msgs.msg import Float32MultiArray


class V2VBridge:
    def __init__(self, channel_profile: str, woc: bool = False):
        self.woc = woc
        params = self._load_params(channel_profile)
        self.mu_delay    = params['mu_delay']
        self.sigma_delay = params['sigma_delay']

        self._buf  = {'ego': deque(maxlen=500), 'target': deque(maxlen=500)}
        self._lock = threading.Lock()
        self._pos  = {'ego': (0.0, 0.0), 'target': (0.0, 0.0)}
        self._perf = {'ego': {'delay': self.mu_delay, 'dropped': True},
                      'target': {'delay': self.mu_delay, 'dropped': True}}

        rospy.init_node('v2v_bridge')

        if not self.woc:
            rospy.Subscriber('/ego/EgoShareInfo',    ShareInfo, lambda m: self._rx('ego',    m))
            rospy.Subscriber('/target/EgoShareInfo', ShareInfo, lambda m: self._rx('target', m))

        self._pub_target      = rospy.Publisher('/target/TargetShareInfo',         ShareInfo,         queue_size=1)
        self._pub_ego         = rospy.Publisher('/ego/TargetShareInfo',             ShareInfo,         queue_size=1)
        self._pub_perf_ego    = rospy.Publisher('/ego/CommunicationPerformance',    Float32MultiArray, queue_size=1)
        self._pub_perf_target = rospy.Publisher('/target/CommunicationPerformance', Float32MultiArray, queue_size=1)

    def _load_params(self, profile: str) -> dict:
        yaml_path = os.path.join(os.path.dirname(__file__), '../config/channel_model.yaml')
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        p = cfg['profiles'][profile]
        self.mu_delay    = p['mu_delay']
        self.sigma_delay = p['sigma_delay']
        self.delay_clip  = p.get('delay_clip', [10, 200])
        self.prr_near    = p.get('prr_near', 0.848)
        self.prr_far     = p.get('prr_far',  0.551)
        self.d50         = p.get('d50',      97.8)
        self.k           = p.get('k',        5.0)
        return p

    def _prr(self, d: float) -> float:
        """3GPP TR 37.885 기반 거리 의존 PRR sigmoid."""
        return self.prr_far + (self.prr_near - self.prr_far) / (1.0 + math.exp((d - self.d50) / self.k))

    def _rx(self, sender: str, msg: ShareInfo):
        self._pos[sender] = (msg.pose.x, msg.pose.y)
        delay_ms = max(self.delay_clip[0], min(self.delay_clip[1], random.gauss(self.mu_delay, self.sigma_delay)))
        with self._lock:
            self._buf[sender].append((time.time(), delay_ms, msg))

    def _deliver(self):
        now = time.time()
        dx = self._pos['ego'][0] - self._pos['target'][0]
        dy = self._pos['ego'][1] - self._pos['target'][1]
        d  = math.sqrt(dx**2 + dy**2)
        prr = self._prr(d)

        for sender, buf in self._buf.items():
            receiver  = 'target' if sender == 'ego' else 'ego'
            pub       = self._pub_target if receiver == 'target' else self._pub_ego
            perf_pub  = self._pub_perf_target if receiver == 'target' else self._pub_perf_ego

            delivered, remaining = [], deque()
            with self._lock:
                for item in buf:
                    send_t, delay_ms, msg = item
                    if now >= send_t + delay_ms / 1000.0:
                        delivered.append((delay_ms, msg))
                    else:
                        remaining.append(item)
                self._buf[sender] = remaining

            if delivered:
                delay_ms, msg = delivered[-1]
                if random.random() < prr:
                    pub.publish(msg)
                    self._perf[sender] = {'delay': delay_ms, 'dropped': False}
                else:
                    self._perf[sender]['dropped'] = True

            ch = self._perf[sender]
            perf_pub.publish(Float32MultiArray(data=[
                1.0 if ch['dropped'] else 0.0,
                prr * 100.0,
                d,
                ch['delay'] if not ch['dropped'] else 0.0,
            ]))

    def spin(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if not self.woc:
                self._deliver()
            rate.sleep()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default='yeongjong',
                        choices=['kiapi', 'yeongjong', 'ideal', 'degraded'])
    parser.add_argument('--woc', action='store_true')
    args, _ = parser.parse_known_args()
    V2VBridge(args.channel, args.woc).spin()


if __name__ == '__main__':
    main()
