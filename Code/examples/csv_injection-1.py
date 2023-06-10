import csv
import itertools
import json
import types
from flask import Response, escape
from werkzeug.contrib.iterio import IterI
import xlsxwriter

def get_formatted_response(format, queryrun, reader, resultset_id):
    if format == 'json':
        return json_formatter(queryrun, reader, resultset_id)
    elif format == 'json-lines':
        return json_line_formatter(reader, resultset_id)
    elif format == 'csv':
        return separated_formatter(reader, resultset_id, ',')
    elif format == 'tsv':
        return separated_formatter(reader, resultset_id, '\t')
    elif format == 'wikitable':
        return wikitable_formatter(reader, resultset_id)
    elif format == 'xlsx':
        return xlsx_formatter(reader, resultset_id)
    elif format == 'html':
        return html_formatter(reader, resultset_id)
    return Response('Bad file format', status=400)

class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, types.GeneratorType):
            try:
                first = next(o)
            except StopIteration:
                return []
            else:
                return type('_FakeList', (list,), {
                    '__iter__': lambda self: itertools.chain((first,), o),
                    '__bool__': lambda self: True
                })()
        elif isinstance(o, bytes):
            return o.decode('utf-8')
        else:
            return super().default(o)

def _join_lines(gen):
    for v in gen:
        yield v
    yield '\n'

def _stringify_results(rows):
    for row in rows:
        r = list(row)
        for i, v in enumerate(r):
            if isinstance(v, bytes):
                r[i] = v.decode('utf-8')
        yield r

class _IterI(IterI):
    def write(self, s):
        if s:
            oldpos = self.pos
            super().write(s)
            if (self.pos) // 1024 > oldpos // 1024:
                self.flush()

def separated_formatter(reader, resultset_id, delim=','):
    rows = _stringify_results(reader.get_rows(resultset_id))
    def respond(stream):
        csvobject = csv.writer(stream, delimiter=delim)
        csvobject.writerows(rows)
    return Response(_IterI(respond), content_type='text/html; charset=utf-8')

def json_line_formatter(reader, resultset_id):
    rows = reader.get_rows(resultset_id)
    def respond():
        headers = None
        for row in rows:
            if headers is None:
                headers = row
            else:
                yield json.dumps(dict(zip(headers, row)), cls=_JSONEncoder, check_circular=False)
    return Response(_join_lines(respond()), mimetype='application/json')

def json_formatter(qrun, reader, resultset_id):
    rows = reader.get_rows(resultset_id)
    header = next(rows)
    data = {
        'meta': {
            'run_id': qrun.id,
            'rev_id': qrun.rev.id,
            'query_id': qrun.rev.query.id,
        },
        'headers': header,
        'rows': rows
    }
    def respond(stream):
        json.dump(data, stream, cls=_JSONEncoder, check_circular=False)
    return Response(_IterI(respond), mimetype='application/json')

def wikitable_formatter(reader, resultset_id):
    rows = _stringify_results(reader.get_rows(resultset_id))
    header = next(rows)
    def respond():
        yield '{| class="wikitable"'
        yield '!' + '!!'.join(map(str, header))
        for row in rows:
            yield '|-'
            yield '|' + '||'.join(map(str, row))
        yield '|}'
    return Response(_join_lines(respond()), content_type='text/plain; charset=utf-8')

def xlsx_formatter(reader, resultset_id):
    rows = _stringify_results(reader.get_rows(resultset_id))
    def respond(stream):
        workbook = xlsxwriter.Workbook(stream, {'constant_memory': True})
        worksheet = workbook.add_worksheet()
        for row_num, row in enumerate(rows):
            for col_num, cell in enumerate(row):
                if (worksheet.write(row_num, col_num, cell) < -1 and isinstance(cell, str)):
                    worksheet.write_string(row_num, col_num, cell)
        workbook.close()
    return Response(_IterI(respond), mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

def html_formatter(reader, resultset_id):
    rows = _stringify_results(reader.get_rows(resultset_id))
    header = next(rows)
    def respond():
        yield '<table>\n'
        yield '<tr>'
        for col in header:
            yield '<th scope="col">%s</th>' % escape(col)
        yield '</tr>\n'
        for row in rows:
            yield '<tr>'
            for col in row:
                yield '<td>%s</td>' % escape(col)
            yield '</tr>\n'
        yield '</table>'
    return Response(_join_lines(respond()), content_type='text/html; charset=utf-8')
