from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import sys
sys.path.insert(0, '/storage/jalverio/sentence-tracker/st')
from st import load_model
import json


class S(BaseHTTPRequestHandler):
    def __init__(self):
        self.model = load_model(robot=True)
        super().__init__()

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
        data = np.array(eval(data.decode("utf-8")['images']))
        print(json.loads(data))


        # self.send_response(200, message='hello')
        # self.end_headers()

        # data = simplejson.loads(self.data_string)
        # # with open("test123456.json", "w") as outfile:
        # #     simplejson.dump(data, outfile)
        # print("{}".format(data))
        # # f = open("for_presen.py")
        # self.wfile.write('this is my response')
        # return


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