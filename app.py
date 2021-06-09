import plotly.express as px
from wordcloud import WordCloud
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import base64
import textwrap
import os
import json
from time import time
from collections import Counter

stop_words = {'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'reporter',
 'reported',
 'stated',
 'from',
 'zzz',
 'flight',
 'aircraft',
 'br',
 '<br>'
}

embed_df = pd.read_csv('data/embeddings_combined.csv')
#Box style options
graph_style = {
    'width': '70vw',
    'height': '60vh',
    'margin': '0px',
    'plot_bgcolor': 'red',
    'paper_bgcolor': 'blue',
}

wide_container = {
    'width': graph_style['width'],
    'border-radius': '10px',
    'background-color': '#f9f9f9',
    'margin': '10px',
    'padding': '15px',
    'box-shadow': '2px 2px 2px 2px lightgrey'
}

thin_container = {
    'width': '25vw',
    'border-radius': '10px',
    'background-color': '#f9f9f9',
    'margin': '10px',
    'padding': '15px',
    'box-shadow': '2px 2px 2px 2px lightgrey'
}

button_style = {
    'width': '10vw',
    'border-radius': '10px',
    'background-color': '#f9f9f9',
    'margin': '10px',
    'padding': '15px',
    'box-shadow': '2px 2px 2px 2px lightgrey'
}

dropdown_style = {
    'width': '20vw',
    'border-radius': '10px',
    'background-color': '#f9f9f9',
    'margin': '10px',
    'padding': '15px',
    'box-shadow': '2px 2px 2px 2px lightgrey'
}

employees = {
    'mroncalli@gmail.com': 'Michael Roncalli',
    'seaurchinfan@gmail.com': 'Graham Morehead',
    'asarmiento@gmail.com': 'Ali Sarmiento',
    'fmcgowan@gmail.com': 'Fitz McGowan',
    'mborfitz@gmail.com': 'Mike Borfitz',
    'nsmith@gmail.com': 'Nick Smith',
    'rswanson@gmail.com': 'Ron Swanson',
    'anneperkins@gmail.com': 'Anne Perkins'
}

def distance_from_centers(df, scale_factor=1):
    '''Calculates point's distance from respective cluster center.'''
    embedding, labels = df[['x', 'y', 'z']], df.Cluster
    centers = df.groupby('Cluster').mean().drop(columns='Severity')

    # Apply scale factor to separate out clusters
    centers *= scale_factor
    centers = np.array(list(map(lambda x: centers.loc[x], labels)))
    return embedding - centers


def recenter_camera(point, dims):
    '''Converts datapoint coordinates from raw point to camera range (0, 0, 0 being pure center).'''
    #     point = x, y, z
    print('point', point)
    map_center = (dims.max() + dims.min()) / 2  # words
    aspect_ratio = 7 / 6  # hard coded
    multiplier = 0.5 * aspect_ratio
    half_length_of_axes = abs(map_center - dims.min())
    print('half: ', half_length_of_axes,
          'map_center: ', map_center, 'aspect_ratio: ', aspect_ratio)
    new_coords = ((point - map_center) / half_length_of_axes) * multiplier

    return dict(new_coords)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dbc.Row([
        # Display single narrative
        dbc.Col([
            html.H1('Select Incident Severity Level: ',
                    style={
                        'margin': '10px',
                        'padding': '15px',
                    }
                    ),
            html.Div([
                # Change this to incident/ACN Numbers
                dcc.Slider(
                    id='severity-slider',
                    min=0,
                    max=100,
                    value=0,
                    step=1,
                    marks={
                        0: '0%',
                        25: '25%',
                        50: '50%',
                        75: '75%',
                        100: '100%'
                    }
                ),
            ], style=wide_container),
            html.Div([dcc.Graph(id='3d-graph', style=graph_style)],
                     ),
            html.Br(),
            dbc.Row([
                html.Button('Rotate Graph', style=button_style),
                html.Button('Focus on Most Severe', id='zoom-button', style=button_style),
                html.Div([
                    html.Div('Find incidents by keyword:'),
                    dcc.Input(
                        id='search-bar',
                    ),
                ], style=dropdown_style
                ),
                html.Div([
                    html.Div('Send incidents to:'),
                    dcc.Dropdown(
                        id='user-dropdown',
                        options=[{'label': val, 'value': key} for key, val in employees.items()],
                    ),
                    html.Div(id='email-selection'),

                ], style=dropdown_style
                ),

            ], style={'margin-left': '5vw'})
        ]),
        # Cluster wordmap
        dbc.Col([
            html.Div([
                html.Img(src=app.get_asset_url('Topologe Logo Alpha+ (1).png'),
                         style={
                             'width': '66%',
                             'margin-left': '5vw'
                         }
                         )
            ]),
            html.Br(),
            html.H3(id='cloud-title', children='Top Words in Selected Cluster'),
            html.Div(html.Img(id='word-cloud', style=thin_container)),
            dcc.Store(id='intermediate-value')
        ])
    ]),
])


@app.callback(
    Output('email-selection', 'children'),
    [Input('user-dropdown', 'value'),
     ]
)
def print_user(user):
    if user is None:
        return
    return 'Sent to: ' + str(user)


@app.callback(
    Output('3d-graph', 'figure'),
    [Input('severity-slider', 'value'),
     Input('zoom-button', 'n_clicks')
     ]
)
def plot_severity(threshold, zoom_clicks):
    '''
    Plots 3d embedding with clusters, label by group severity rating.
        Additional idea - toggle color by severity max.
    Args:
        x - 3d embedding
        y - labels
        hover - optional hover data added to figure.
        threshold - only show clusters with higher average than threshold.
    '''
    start = time()
    # Filter out noise points and clusters below threshold
    clustered = (embed_df.Cluster >= 0) & (embed_df.Severity >= threshold)

    filtered_df = embed_df[clustered]

    # Record which cluster has highest Severity score
    most_severe = filtered_df.groupby('Cluster').mean()['Severity'].argmax()

    # Apply scale factor
    filtered_df[['x', 'y', 'z']] = distance_from_centers(
        filtered_df, scale_factor=4
    )
    dims = filtered_df[['x', 'y', 'z']]

    fig = px.scatter_3d(
        filtered_df, x='x', y='y', z='z',
        color='Severity',
        color_continuous_scale='rdylgn_r',
        title='NTSB Incident Topics',
        hover_data={
            'x': True,
            'y': True,
            'z': True,
            'Cluster': True,
            'Severity': False,
            'Synopsis': True,
        }
    )
    #     fig.update_traces(hovertemplate = '%{text}',
    #         text = filtered_df.Synopsis
    #                      )
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5

    fig.update_traces(
        hovertemplate='%{text} <extra></extra>`',
        text=filtered_df.Synopsis
    )

    # Rotation button
    fig.update_layout(
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
    #          updatemenus=[dict(type='buttons',
    #                   showactive=False,
    #                   y=0,
    #                   x=0,
    #                   xanchor='left',
    #                   yanchor='bottom',
    #                   pad=dict(t=45, r=10),
    #                   buttons=[dict(label='Rotate',
    #                                  method='animate',
    #                                  args=[None, dict(frame=dict(duration=5, redraw=True),
    #                                                              transition=dict(duration=0),
    #                                                              fromcurrent=True,
    #                                                              mode='immediate'
    #                                                             )]
    #                                             )
    #                                       ]
    #                               )
    #                         ]
)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                showticklabels=False,
            )
        ),
    )

    if zoom_clicks is not None:
        #         print('click counter: ', zoom_clicks)
        # Provide clusters in descending order of highest Severity score
        center = embed_df.groupby('Cluster').mean().sort_values(
            by='Severity', ascending=False
        ).drop(columns='Severity').iloc[zoom_clicks - 1]
        fig.update_layout(
            #             scene_camera_eye = dict(x=1, y=1, z=0.8),
            scene_camera_center=recenter_camera(center, dims)
        )

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    #     frames=[]
    #     for t in np.arange(0, 6.26, 0.05):
    #         xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    #         frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    #     fig.frames=frames

    end = time()
    print(f'Time elapsed: {round(end - start, 4)} seconds.')
    return fig

@app.callback(
    Output('intermediate-value', 'value'),
    [Input('3d-graph', 'clickData')]
)
def prepare_cloud(click_data):
    return json.dumps(click_data)

@app.callback(
    Output('word-cloud', 'src'),
    [Input('intermediate-value', 'value')]
)
def generate_cloud(json_data):
    json_data = json.loads(json_data)
    print('JSON Data: ', json_data)
    #Create wordcloud out of first cluster if no cluster provided
    if json_data is None:
        cluster_number = 1
    else:
        cluster_number = json_data['points'][0]['customdata'][0]
        print('Showing Cluster: ', cluster_number)

    data = embed_df[embed_df.Cluster == cluster_number]['Synopsis']

    # Compile string of all words in corpus
    word_bag = ''.join(y for y in [x for x in data.str.lower().values])
    word_counts = Counter(word_bag.split())
    final_counts = {}
    #Create loop that captures first 20 terms not in stop_words
    building = True
    while building:
        for a, b in list(sorted(word_counts.items(), key=lambda item: item[1], reverse=True)):
            if a not in stop_words:
                final_counts.update({a: b})
                if len(final_counts) >= 20:
                    building = False
                    break

    wc = WordCloud(
        background_color = '#f9f9f9',
        # relative_scaling=1,
    ).generate_from_frequencies(final_counts)
    wc_img = wc.to_image()
    with BytesIO() as buffer:
        wc_img.save(buffer, 'png')
        img2 = base64.b64encode(buffer.getvalue()).decode()

    return "data:image/png;base64," + img2


if __name__ == '__main__':
    app.run_server(port=8099)