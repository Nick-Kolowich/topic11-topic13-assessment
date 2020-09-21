{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import scipy.stats as stats\n",
    "import seaborn as sns\n",
    "import csv\n",
    "import json\n",
    "import requests\n",
    "from scipy.stats import kurtosis, skew"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Growth rate annualized</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1947-04-01</td>\n",
       "      <td>-1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1947-07-01</td>\n",
       "      <td>-0.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1947-10-01</td>\n",
       "      <td>6.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1948-01-01</td>\n",
       "      <td>6.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1948-04-01</td>\n",
       "      <td>6.8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Date  Growth rate annualized\n",
       "0 1947-04-01                    -1.0\n",
       "1 1947-07-01                    -0.8\n",
       "2 1947-10-01                     6.4\n",
       "3 1948-01-01                     6.2\n",
       "4 1948-04-01                     6.8"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from fredapi import Fred\n",
    "fred = Fred(api_key_file=(r'C:\\Users\\Nick\\Downloads\\assessment\\api_key_file.txt'))\n",
    "GDP = fred.get_series('A191RL1Q225SBEA')\n",
    "GDP_df = GDP.reset_index()\n",
    "GDP_df.columns = ['Date','Growth rate annualized']\n",
    "GDP_df['Date'] = pd.to_datetime(GDP_df['Date'])\n",
    "GDP = GDP_df[(GDP_df['Date'] < '2020-01-01')]\n",
    "GDP.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "mu = round(GDP['Growth rate annualized'].mean(), 4)\n",
    "sigma = round(GDP['Growth rate annualized'].std(), 4)\n",
    "n = GDP['Growth rate annualized'].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Growth rate z-score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1947-04-01</td>\n",
       "      <td>-1.097551</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1947-07-01</td>\n",
       "      <td>-1.045385</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1947-10-01</td>\n",
       "      <td>0.832599</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1948-01-01</td>\n",
       "      <td>0.780432</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1948-04-01</td>\n",
       "      <td>0.936931</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Date  Growth rate z-score\n",
       "0 1947-04-01            -1.097551\n",
       "1 1947-07-01            -1.045385\n",
       "2 1947-10-01             0.832599\n",
       "3 1948-01-01             0.780432\n",
       "4 1948-04-01             0.936931"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "GDP_z = pd.DataFrame(GDP['Growth rate annualized'] - mu)/(sigma)\n",
    "GDP_z.reset_index()\n",
    "GDP_z.columns = ['Growth rate z-score']\n",
    "GDP_z['Date'] = GDP_df['Date']\n",
    "z_scores = GDP_z['Growth rate z-score']\n",
    "GDP_z = GDP_z[['Date','Growth rate z-score']]\n",
    "GDP_z.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "         Date  Growth rate z-score\n",
      "11 1950-01-01             3.519158\n"
     ]
    }
   ],
   "source": [
    "Q1_1950 = GDP_z.loc[GDP_z['Date'] == '1950-01-01']\n",
    "print(Q1_1950)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.2079"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.8339"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sigma"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "291"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2170e98d160>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(GDP['Growth rate annualized'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2170f116b20>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# z value dist plotted\n",
    "z_data = [(x - GDP['Growth rate annualized'].mean())/GDP['Growth rate annualized'].std() for x in GDP['Growth rate annualized']]\n",
    "sns.distplot(z_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Frequency\n",
    "stats_df = GDP.groupby('Growth rate annualized')['Growth rate annualized'].agg('count').pipe(pd.DataFrame)\n",
    "\n",
    "stats_df.columns = ['frequency']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>frequency</th>\n",
       "      <th>pdf</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Growth rate annualized</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>-10.0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>-8.4</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>-8.0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>-6.1</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>-5.9</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11.9</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12.8</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13.8</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16.4</th>\n",
       "      <td>2</td>\n",
       "      <td>0.006873</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16.7</th>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>123 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                        frequency       pdf\n",
       "Growth rate annualized                     \n",
       "-10.0                           1  0.003436\n",
       "-8.4                            1  0.003436\n",
       "-8.0                            1  0.003436\n",
       "-6.1                            1  0.003436\n",
       "-5.9                            1  0.003436\n",
       "...                           ...       ...\n",
       " 11.9                           1  0.003436\n",
       " 12.8                           1  0.003436\n",
       " 13.8                           1  0.003436\n",
       " 16.4                           2  0.006873\n",
       " 16.7                           1  0.003436\n",
       "\n",
       "[123 rows x 2 columns]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# PDF\n",
    "stats_df['pdf'] = (stats_df['frequency']) / (sum(stats_df['frequency']))\n",
    "stats_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Growth rate annualized</th>\n",
       "      <th>frequency</th>\n",
       "      <th>pdf</th>\n",
       "      <th>cdf</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-10.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>0.003436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-8.4</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>0.006873</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-8.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>0.010309</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-6.1</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>0.013746</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-5.9</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>0.017182</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>118</th>\n",
       "      <td>11.9</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>0.982818</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>119</th>\n",
       "      <td>12.8</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>0.986254</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>120</th>\n",
       "      <td>13.8</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>0.989691</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>121</th>\n",
       "      <td>16.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0.006873</td>\n",
       "      <td>0.996564</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>122</th>\n",
       "      <td>16.7</td>\n",
       "      <td>1</td>\n",
       "      <td>0.003436</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>123 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Growth rate annualized  frequency       pdf       cdf\n",
       "0                     -10.0          1  0.003436  0.003436\n",
       "1                      -8.4          1  0.003436  0.006873\n",
       "2                      -8.0          1  0.003436  0.010309\n",
       "3                      -6.1          1  0.003436  0.013746\n",
       "4                      -5.9          1  0.003436  0.017182\n",
       "..                      ...        ...       ...       ...\n",
       "118                    11.9          1  0.003436  0.982818\n",
       "119                    12.8          1  0.003436  0.986254\n",
       "120                    13.8          1  0.003436  0.989691\n",
       "121                    16.4          2  0.006873  0.996564\n",
       "122                    16.7          1  0.003436  1.000000\n",
       "\n",
       "[123 rows x 4 columns]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# CDF\n",
    "stats_df['cdf'] = stats_df['pdf'].cumsum()\n",
    "stats_df = stats_df.reset_index()\n",
    "stats_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "pdf_plot = stats_df.plot.bar(x = 'Growth rate annualized', y = 'pdf', grid = True)\n",
    "pdf_plot.set_xticks(ticks=[])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2170f497850>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "stats_df.sort_values('Growth rate annualized').plot(x = 'Growth rate annualized', y = 'cdf', grid = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAD1CAYAAABA+A6aAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAUOUlEQVR4nO3df7BcZX3H8fe3IRBs0qskeEcT5KZt1IkGxRtBqy2JWBtwBFuCkND8YIZJYYzOdEZLGDvK4DijBesgBCMiQ3Got/izETKNthRFflgSGxMig80E1BucKommghMx+O0fexI3y2Z3b+7eu7tn36+Znfuc8zz3Oc+5yXzus885Z29kJpKk3vd7nR6AJKk9DHRJKgkDXZJKwkCXpJIw0CWpJI7r1IFnzZqVQ0NDnTq8JPWkrVu3PpWZJ9er61igDw0NsWXLlk4dXpJ6UkT88Gh1LrlIUkkY6JJUEga6JJVEx9bQ6/nNb37D6OgoBw4c6PRQxm3atGnMmTOHqVOndnookvpEVwX66OgoM2bMYGhoiIjo9HCOWWayd+9eRkdHmTt3bqeHI6lPdNWSy4EDB5g5c2ZPhzlARDBz5sxSvNOQ1DuaBnpE3BoRP42IR45SHxHxyYjYFRHbI+J14xlQr4f5IWU5D0m9o5UZ+m3Akgb15wDzitca4FPjH5YkaayarqFn5rciYqhBk/OB27PyweoPRcQLI+IlmfmT8Q5uaN3d4+3iCE989O1t6Wf69Ok8/fTTALz//e9n06ZNnHvuuVx77bVt6V+SjkU7LorOBn5ctT1a7HteoEfEGiqzeAYHB7n33nuPqB8YGOCXv/xlG4ZUXzv7PtTXpz/9aXbv3s0JJ5zwvP4PHDjwvHOUpFbt2LMfgAWzB9ixZz8LZg80bN+OQK+3WFz3zyBl5s3AzQALFy7MRYsWHVH/6KOPMmPGjDYMqb5W+7799tu57rrriAhOO+00rrnmGpYvX87BgwdZsmTJ4b7OO+88nnnmGd761rdy1VVXcdFFFx3Rz7Rp0zj99NPbfh6S+sPqYpXiiUsWsXrd3TxxyaKG7dsR6KPAKVXbc4An29BvR+zcuZOPfOQj3H///cyaNYt9+/axevVqrrjiClauXMn69esPt924cSPTp09n27ZtHRyxpLIZWnf3MS0Rt+O2xY3AyuJulzcA+9uxft4p99xzD0uXLmXWrFkAnHTSSdx///0sW7YMgBUrVnRyeJJKpPo6YTuuGTadoUfE54FFwKyIGAU+BEwFyMwNwCbgXGAX8Cvg0nGPqoMys+4th96GKKkdDgV3u27SqNZ0hp6ZyzLzJZk5NTPnZOZnM3NDEeZkxbsz848yc0Fm9vRn4p599tnceeed7N27F4B9+/bxpje9iZGREQDuuOOOTg5PUg9p9wy8ma569L/WRPwGa+ZVr3oVH/jABzjrrLOYMmUKp59+Otdffz3Lly/n+uuv54ILLpj0MUnqHRM5A2+mqwO9U1atWsWqVauO2Pfggw8eLq9bt+5w+dD96JL6VydDvFpXfZaLJHW7yV5GGQsDXZKaGFp3d9eFdz1dF+iVTxDofWU5D6mf9UKIV+uqQJ82bRp79+7t+TA89Hno06ZN6/RQJPWRrrooOmfOHEZHR/nZz37W6aGM26G/WCRJk6WrAn3q1Kn+hR9Jk676Uftjfey+G3TVkoskTZZeudA5Fga6pNLr5lsN28lAl1RKZZyBN2OgSyqVfgvxaga6pJ7XzyFezUCXpJIw0CX1JGflz2egS+oZ/XihcywMdEkqCQNdUtdzVt4aA11SVzLEx85Al9Q1XCMfHwNdUkcZ4u1joEuaNP3ymSqdYqBLUkkY6JImlEsqk8dAl6SSMNAlqSQMdEkTwmWWyWegS1JJGOiS2sZZeWcZ6JJUEga6pHFxVt49DHRJY+a95d2ppUCPiCUR8VhE7IqIdXXqByLiaxHxvYjYGRGXtn+okjrNEO9uTQM9IqYA64FzgPnAsoiYX9Ps3cD3M/M1wCLg4xFxfJvHKklqoJUZ+hnArszcnZnPAiPA+TVtEpgREQFMB/YBB9s6Ukkd4ay8d0RmNm4QsRRYkpmXFdsrgDMzc21VmxnARuCVwAzgosx83v+CiFgDrAEYHBwcHhkZadd5SJogO/bsZ8HsgSPKO/bsBzhcrq0fS9va+rG0bedxe+UcFi9evDUzF9b9x8rMhi/gQuCWqu0VwA01bZYCnwAC+GPgceAPGvU7PDyckrrLqVfedfhrdXms9ePpq1PH7ZVzALbkUXK1lSWXUeCUqu05wJM1bS4FvlyMZVcR6K9soW9JUpu0EugPA/MiYm5xofNiKssr1X4EnA0QEYPAK4Dd7RyoJKmx45o1yMyDEbEW2AxMAW7NzJ0RcXlRvwH4MHBbROygsuxyZWY+NYHjliTVaBroAJm5CdhUs29DVflJ4G3tHZokaSx8UlTqc96WWB4GuiSVhIEuSSVhoEt9ovoDtVxmKScDXZJKwkCXSs7ZeP8w0CWpJAx0qYSclfcnA10qCf+KkAx0qYcZ4qpmoEtSSRjoUg9yVq56DHSpRxjiasZAl6SSMNClLuZFT42FgS5JJWGgS1JJGOiSVBIGutSFXDfXsTDQpS5hiGu8DHRJKgkDXeogb0tUOxnoklQSBroklYSBLk0yl1k0UQx0aZIY4ppoBroklYSBLkklYaBLUkkY6NIEct1ck8lAl6SSMNAlqSRaCvSIWBIRj0XErohYd5Q2iyJiW0TsjIhvtneYkqRmjmvWICKmAOuBPwdGgYcjYmNmfr+qzQuBm4AlmfmjiHjxRA1YklRfKzP0M4Bdmbk7M58FRoDza9osB76cmT8CyMyftneYkqRmWgn02cCPq7ZHi33VXg68KCLujYitEbGyXQOUeo2P9qtTIjMbN4i4EPiLzLys2F4BnJGZ76lqcyOwEDgbOBF4EHh7Zv6gpq81wBqAwcHB4ZGRkTaeitQdduzZD8CC2QPs2LOfBbMHDu8/tK+6fixtJ7Kv8Ry3nX15Do3bLl68eGtmLqSezGz4At4IbK7avgq4qqbNOuDqqu3PAhc26nd4eDilMjn1yrsOf60uN6sfS9uJ7Gs8x21nX55D47bAljxKrray5PIwMC8i5kbE8cDFwMaaNv8K/GlEHBcRLwDOBB5toW9JUps0vcslMw9GxFpgMzAFuDUzd0bE5UX9hsx8NCL+DdgO/Ba4JTMfmciBS5KO1DTQATJzE7CpZt+Gmu1rgWvbNzRJ0lj4pKg0Dt7Nom5ioEtSSRjoklQSBroklYSBLo2R6+bqVga61AIf51cvMNAlqSQMdEkqCQNdasBlFvUSA12SSsJAl6SSMNClGi6zqFcZ6JJUEga6JJWEgS7hg0MqBwNdkkrCQJekkjDQJakkDHRJKgkDXZJKwkBX3/LOFpWNgS5JJWGgS1JJGOiSVBIGuvqO6+YqKwNdkkrCQJekkjDQJakkDHT1BdfN1Q8MdEkqCQNdkkrCQJekkjDQVVp+Vov6TUuBHhFLIuKxiNgVEesatHt9RDwXEUvbN0RJUiuaBnpETAHWA+cA84FlETH/KO0+Bmxu9yClsXBWrn7Vygz9DGBXZu7OzGeBEeD8Ou3eA3wJ+GkbxydJalFkZuMGleWTJZl5WbG9AjgzM9dWtZkN/DPwFuCzwF2Z+cU6fa0B1gAMDg4Oj4yMtOs8pMN27NnPgtkD7NizH+BwecHsgYb1Y2lbW9+p43bLOXTLz6MfzmHx4sVbM3Mh9WRmwxdwIXBL1fYK4IaaNl8A3lCUbwOWNut3eHg4pYlw6pV3Hf5aXW5WP5a2tfWdOm63nEO3/Dz64RyALXmUXG1lyWUUOKVqew7wZE2bhcBIRDwBLAVuioh3ttC31Baum0twXAttHgbmRcRcYA9wMbC8ukFmzj1UjojbqCy5fLWN45QkNdE00DPzYESspXL3yhTg1szcGRGXF/UbJniMkqQWtDJDJzM3AZtq9tUN8sxcPf5hSZLGyidFJakkDHT1LB/tl45koEtSSRjoklQSBroklYSBLkklYaCrp3ghVDo6A12SSsJAV09wVi41Z6BLUkkY6OpazsqlsTHQ1VW86CkdOwNdkkrCQFdXcFYujZ+BLkklYaCrY5yVS+1loGtSGeLSxDHQJakkDHRJKgkDXZJKwkDXhPNhIWlyGOiSVBIGuiaMs3JpchnoklQSBroklYSBLkklYaCrrVw3lzrHQJekkjDQNW7eZy51BwNdx8QQl7qPgS5JJWGga0yclUvdq6VAj4glEfFYROyKiHV16i+JiO3F64GIeE37hypJaqRpoEfEFGA9cA4wH1gWEfNrmj0OnJWZpwEfBm5u90AlSY21MkM/A9iVmbsz81lgBDi/ukFmPpCZPy82HwLmtHeY6iSXWaTeEJnZuEHEUmBJZl5WbK8AzszMtUdp/z7glYfa19StAdYADA4ODo+MjIxz+JooO/bsB2DB7AF27NnPgtkDh/cf2lddP5a2tfVjadvO43oO7TuHbvl59MM5LF68eGtmLqSezGz4Ai4EbqnaXgHccJS2i4FHgZnN+h0eHk51n1OvvOvw1+pys/qxtK2t79RxPYf2nUO3/Dz64RyALXmUXD2ubsofaRQ4pWp7DvBkbaOIOA24BTgnM/e20K8kqY1aWUN/GJgXEXMj4njgYmBjdYOIeBnwZWBFZv6g/cOUJDXTNNAz8yCwFthMZTnlzszcGRGXR8TlRbMPAjOBmyJiW0RsmbARq+286CmVQytLLmTmJmBTzb4NVeXLgOddBJUkTR6fFJWkkjDQ+5QfriWVj4EuSSVhoPcRZ+VSuRnofcAQl/qDgS5JJWGgl5Szcqn/GOglcCi8XSOX+puB3qMMbkm1DPQeYohLasRA73Iuo0hqlYHepQxxSWNloHcRQ1zSeBjoHeaSiqR2MdA7wBCXNBEM9ElkiEuaSAa6JJWEgT7BnJVLmiwG+gRwjVxSJxjoklQSBnobOSuX1EkGuiSVhIE+Ts7KJXULA/0YeNFTUjcy0Jvwj0dI6hUGeh0Gt6ReZKAXnIFL6nV9GejVwW2ISyqLvgl0g1tS2ZUy0L2QKakflSbQDW5J/a6nA90ZuCT9Ts8EussoktRYS4EeEUsi4rGI2BUR6+rUR0R8sqjfHhGva8fgDG5Jal3TQI+IKcB64BxgPrAsIubXNDsHmFe81gCfGutAnIFL0vi0MkM/A9iVmbsz81lgBDi/ps35wO1Z8RDwwoh4SbOODW5Jap/IzMYNIpYCSzLzsmJ7BXBmZq6tanMX8NHM/Hax/R/AlZm5paavNVRm8ACvAB5r14lIUp84NTNPrldxXAvfHHX21f4WaKUNmXkzcHMLx5QkjVErSy6jwClV23OAJ4+hjSRpArUS6A8D8yJibkQcD1wMbKxpsxFYWdzt8gZgf2b+pM1jlSQ10HTJJTMPRsRaYDMwBbg1M3dGxOVF/QZgE3AusAv4FXDpxA1ZklRP04uiUrWIGAQ+AbwB+DnwLPAPmfmVNh7jtcBLM3NTsX018HRmXtem/lcDX8/MnlgWjIgh4K7MfHVELARWZuZ7x9nnbUWfXxz/CNUteuZJUXVeRATwVeBbmfmHmTlMZQluTp22rVxwP5rXUnnHd8yK5yeOZjXw0vH03ymZuWW8Ya7yMtA1Fm8Bni2W2QDIzB9m5g1QmflGxBci4mvA1yPipIj4avH08EMRcVrRbkdEvLC45rI3IlYW+z8XEW8DrgEuiohtEXFRcaj5EXFvROyOiLqBFhFPR8Q1EfEd4I0R8cGIeDgiHomIm4vjLQUWAncU/Z8YEcMR8c2I2BoRm+s9QxER74iI70TEf0fEvxfvVIiIqyPi1tqxRcRQRDwaEZ+JiJ0R8fWIOLGou7eYaRMRsyLiiarvuS8ivlu8/qTOOBYVtwkTEZuKc9gWEfsjYlVETImIa4vz3h4Rf1O0jYi4MSK+HxF3Ay8eyz+8ekRm+vLV0gt4L/CJBvWrqdzxdFKxfQPwoaL8FmBbUd4AvB14NZWL7p8p9v8PML3o58aqfq8GHgBOAGYBe4GpdY6fwLuqtk+qKn8OeEdRvhdYWJSnFn2fXGxfROU6UW3fL+J3S5SXAR9vNDZgCDgIvLZodyfw13WOPwt4oii/AJhWlOcBW4ryEPBIUV5EZamkemzDwHZggMpzHn9f7D8B2ALMBf4K+AaV62AvBX4BLO30/ylf7X2N522x+lxErAfeTGXW/vpi9zcyc19RfjNwAUBm3hMRMyNiALgP+DPgh1Q+JmJNRMwG9mXm05WVnee5OzN/Dfw6In4KDFL55VHtOeBLVduLI+LvqATlScBO4Gs13/MKKr9YvlEcdwpQ7w6tOcC/FLP344HHm4wN4PHM3FaUt1IJ5kamAjcW1xCeA17epD0RMYvKL6t3Zeb+4h3OacU7EaiE/DwqP+/PZ+ZzwJMRcU+zvtV7DHSNxU6KgAbIzHcXgVL9RPAzVeWjPXD2LeDdwMuADwB/CSylEvRH8+uq8nPU/797oAgsImIacBOVmfCPiwur0+p8TwA7M/ONDY4NlXcb/5iZGyNiEZWZebOx1e4/sSgf5HfLndVj+lvgf4HXFPUHGg2ouE4wAlyTmY9Unc97MnNzTdtzqfOwn8rFNXSNxT3AtIi4omrfCxq0/xZwCVTWfoGnMvP/MvPHVJYa5mXmbuDbwPv4XaD/EpgxzrEeCsqnImI6lV8Yh1T3/xhwckS8sRjn1Ih4VZ3+BoA9RXnVOMf2BJVlEmrGNQD8JDN/C6yg8m6hkY8C2zNzpGrfZuCKiJgKEBEvj4jfp/JvcXGxxv4SYPE4z0FdyEBXyzIzgXcCZ0XE4xHxX8A/AVce5VuuBhZGxHYq4VMdhN8BflCU7wNmUwl2gP+kchG0+qLoWMf6C+AzwA4qd+Y8XFV9G7AhIrZRCc2lwMci4nvANuB5FyOLc/lCRNwHPHUsY6pyHZXQfYDKL7ZDbgJWRcRDVJZbnqn3zVXeB7yt6sLoecAtwPeB70bEI8Cnqbxj+AqVaxQ7qCxzfXOc56Au5H3oklQSztAlqSQMdEkqCQNdkkrCQJekkjDQJakkDHRJKgkDXZJK4v8B38BpCdnuTCQAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "cdf_bar = stats_df.plot.bar(x = 'Growth rate annualized', y = 'cdf', grid = True)\n",
    "cdf_bar.set_xticks(ticks=[])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Level Up"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Find the z-score of a 6.1% annual growth (China's 2019 GDP growth rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7543\n"
     ]
    }
   ],
   "source": [
    "z_score = (6.1 - mu) / (sigma)\n",
    "print(round(z_score, 4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Skew is: 0.1352\n",
      "Kurtosis is: 1.6133\n"
     ]
    }
   ],
   "source": [
    "skew = stats.skew(GDP['Growth rate annualized'])\n",
    "kurtosis = stats.kurtosis(GDP['Growth rate annualized'])\n",
    "print(\"Skew is: \" + str(round(skew, 4)))\n",
    "print(\"Kurtosis is: \" + str((round(kurtosis, 4))))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "        1 file(s) moved.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[NbConvertApp] Converting notebook assessment.ipynb to markdown\n",
      "[NbConvertApp] Support files will be in assessment_files\\\n",
      "[NbConvertApp] Making directory assessment_files\n",
      "[NbConvertApp] Making directory assessment_files\n",
      "[NbConvertApp] Making directory assessment_files\n",
      "[NbConvertApp] Making directory assessment_files\n",
      "[NbConvertApp] Making directory assessment_files\n",
      "[NbConvertApp] Writing 8889 bytes to assessment.md\n"
     ]
    }
   ],
   "source": [
    "!jupyter nbconvert --to markdown assessment.ipynb && move assessment.md README.md"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": false,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {
    "height": "calc(100% - 180px)",
    "left": "10px",
    "top": "150px",
    "width": "349.091px"
   },
   "toc_section_display": true,
   "toc_window_display": true
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
