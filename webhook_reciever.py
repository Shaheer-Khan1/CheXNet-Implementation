import http.server
import socketserver
import json
import datetime
from urllib.parse import urlparse, parse_qs

# Port to listen on (you can change this if needed)
PORT = 8000

class WebhookHandler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
    def do_GET(self):
        """Handle GET requests to show a simple status page"""
        self._set_headers()
        self.wfile.write(json.dumps({
            "status": "Webhook receiver is running",
            "message": "Send POST requests to this URL"
        }).encode())
        
    def do_POST(self):
        """Handle POST requests (the actual webhooks)"""
        # Get the content length
        content_length = int(self.headers['Content-Length'])
        
        # Read the request body
        post_data = self.rfile.read(content_length)
        
        # Parse query parameters from URL if any
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        # Try to parse as JSON
        try:
            json_data = json.loads(post_data.decode('utf-8'))
            data_to_display = json_data
        except:
            # If not JSON, show as plain text
            data_to_display = post_data.decode('utf-8')
        
        # Print received data to console
        print("\n" + "="*50)
        print(f"⏰ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📌 Received webhook at: {self.path}")
        print(f"📋 Headers: {self.headers}")
        print(f"📦 Data: {json.dumps(data_to_display, indent=2)}")
        print("="*50 + "\n")
        
        # Send response
        self._set_headers()
        self.wfile.write(json.dumps({
            "status": "success",
            "message": "Webhook received successfully"
        }).encode())

def run_server():
    handler = WebhookHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n✅ Webhook receiver is running on port {PORT}")
        print(f"📣 Use this URL as your callback URL in CheXNet: http://localhost:{PORT}/webhook")
        print("📝 Press Ctrl+C to stop the server")
        print("\n⏳ Waiting for incoming webhooks...\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down webhook receiver")
            httpd.server_close()

if __name__ == "__main__":
    run_server()