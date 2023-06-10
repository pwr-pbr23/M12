from flask import Flask, redirect, request, render_template_string, abort

app = Flask(__name__)

ALLOWED_DOMAINS = ['example.com', 'trusteddomain.com']

@app.route('/')
def index():
    return render_template_string('''
        <h1>Welcome to the Example App!</h1>
        <p>Click <a href="/redirect?url=http://www.example.com">here</a> to visit our trusted website.</p>
    ''')

@app.route('/redirect')
def redirect_to_url():
    redirect_url = request.args.get('url')
    if redirect_url and is_safe_url(redirect_url):
        return redirect(redirect_url)
    else:
        abort(400)

def is_safe_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc in ALLOWED_DOMAINS

if __name__ == '__main__':
    app.run()
