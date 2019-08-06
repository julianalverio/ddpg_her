from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import sys
sys.path.insert(0, '/storage/jalverio/sentence-tracker/st')
from st import load_model
import json

model = load_model(robot=True)


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        f = open("index.html", "r")
        self.wfile.write(f.read())

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        self._set_headers()
        data = self.rfile.read(int(self.headers['Content-Length']))
        frames = np.array(json.loads(data)['images'])
        try:
            result = model.viterbi_given_frames('The robot picked up the cube', frames)
        except:
            self.send_response(200, message='-1')
            return

        threshold = -10000
        if np.any(result.results[-1].final_state_likelihoods < threshold):
            self.send_response(200, message='0')
        else:
            state = np.argmax(result.results[-1].final_state_likelihoods)
            num_states = result.results[-1].num_states
            reward = state / (num_states - 1)
            self.send_response(200, message=reward)
        # self.end_headers()
        # self.wfile.write('this is my response')


def run(server_class=HTTPServer, handler_class=S, port=500):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

if len(argv) == 2:
    run(port=int(argv[1]))
else:
    run()