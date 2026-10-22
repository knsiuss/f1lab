# -*- coding: utf-8 -*-
"""Tests for the live feed readers in :mod:`intervals`.

The OpenF1 client is replaced with a fake dispatcher, so the tests never touch
the network and pin the shape the pages rely on.
"""

import pandas as pd
import pytest

import intervals
import openf1_client


DRIVERS = [
    {'driver_number': 1, 'full_name': 'Max Verstappen', 'session_key': 11},
    {'driver_number': 4, 'full_name': 'Lando Norris', 'session_key': 11},
    {'driver_number': 16, 'full_name': 'Charles Leclerc', 'session_key': 11},
]


def _fake_client(payloads):
    """Replace ``cached_openf1`` with a dispatcher over fixed payloads."""
    calls = []

    def _cached(endpoint, **params):
        calls.append((endpoint, params))
        data = payloads.get(endpoint, [])
        if isinstance(data, Exception):
            raise data
        return data

    return _cached, calls


@pytest.fixture
def gaps_payload():
    return [
        {'driver_number': 1, 'date': '2026-10-11T12:00:05+00:00',
         'gap_to_leader': 0.0, 'interval': 0.0},
        {'driver_number': 4, 'date': '2026-10-11T12:00:05+00:00',
         'gap_to_leader': 1.5, 'interval': 1.5},
        {'driver_number': 16, 'date': '2026-10-11T12:00:05+00:00',
         'gap_to_leader': None, 'interval': None},
        {'driver_number': 4, 'date': '2026-10-11T12:00:15+00:00',
         'gap_to_leader': 2.5, 'interval': 1.0},
        {'driver_number': 1, 'date': '2026-10-11T12:00:15+00:00',
         'gap_to_leader': 0.0, 'interval': 0.0},
        {'driver_number': 16, 'date': '2026-10-11T12:00:15+00:00',
         'gap_to_leader': 9.25, 'interval': 'PL'},
    ]


class TestGapHistory:
    def test_reads_and_names_every_row(self, monkeypatch, gaps_payload):
        fake, calls = _fake_client({'drivers': DRIVERS, 'intervals': gaps_payload})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        frame = intervals.gap_history(11)

        assert list(frame.columns) == intervals.GAP_COLUMNS
        # The row with neither a gap nor an interval carries no information.
        assert len(frame) == 5
        assert set(frame['Driver']) == {'Max Verstappen', 'Lando Norris',
                                       'Charles Leclerc'}
        assert frame['Date'].is_monotonic_increasing
        assert str(frame['Date'].dt.tz) == 'UTC'
        assert [endpoint for endpoint, _ in calls] == ['intervals', 'drivers']
        assert all(params['session_key'] == 11 for _, params in calls)

    def test_blank_interval_becomes_missing_not_zero(self, monkeypatch, gaps_payload):
        fake, _ = _fake_client({'drivers': DRIVERS, 'intervals': gaps_payload})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        frame = intervals.gap_history(11)
        leclerc = frame[frame['Driver'] == 'Charles Leclerc']

        assert leclerc['GapToLeader'].tolist() == [9.25]
        assert leclerc['Interval'].isna().all()

    def test_empty_feed_gives_an_empty_frame(self, monkeypatch):
        fake, _ = _fake_client({'drivers': DRIVERS, 'intervals': []})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        frame = intervals.gap_history(11)

        assert frame.empty
        assert list(frame.columns) == intervals.GAP_COLUMNS

    def test_a_failing_endpoint_is_not_an_error(self, monkeypatch):
        fake, _ = _fake_client({'drivers': DRIVERS,
                                'intervals': RuntimeError('boom')})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        assert intervals.gap_history(11).empty


class TestLatestGaps:
    def test_keeps_the_most_recent_row_per_driver(self, monkeypatch, gaps_payload):
        fake, _ = _fake_client({'drivers': DRIVERS, 'intervals': gaps_payload})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        latest = intervals.latest_gaps(11)

        assert latest['Driver'].tolist() == ['Max Verstappen', 'Lando Norris',
                                            'Charles Leclerc']
        assert latest['GapToLeader'].tolist() == [0.0, 2.5, 9.25]


class TestGapSeries:
    def test_limits_the_lines_and_pivots_on_time(self, monkeypatch, gaps_payload):
        fake, _ = _fake_client({'drivers': DRIVERS, 'intervals': gaps_payload})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        pivot = intervals.gap_series(11, max_drivers=2)

        assert 'Date' in pivot.columns
        # Only the two closest cars are kept.
        assert set(pivot.columns) == {'Date', 'Max Verstappen', 'Lando Norris'}
        assert len(pivot) == 2
        assert pivot['Lando Norris'].tolist() == [1.5, 2.5]

    def test_no_feed_gives_an_empty_frame(self, monkeypatch):
        fake, _ = _fake_client({})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        assert intervals.gap_series(11).empty


class TestGapSummary:
    def test_counts_the_feed(self, monkeypatch, gaps_payload):
        fake, _ = _fake_client({'drivers': DRIVERS, 'intervals': gaps_payload})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        summary = intervals.gap_summary(11)

        assert summary['rows'] == 5
        assert summary['drivers'] == 3
        assert summary['minutes'] == pytest.approx(0.2)
        assert summary['leader'] == 'Max Verstappen'
        assert summary['last_update'] is not None

    def test_empty_feed_is_all_zeroes(self, monkeypatch):
        fake, _ = _fake_client({})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        summary = intervals.gap_summary(11)

        assert summary == {'rows': 0, 'drivers': 0, 'minutes': 0.0,
                           'last_update': None, 'leader': None}


class TestRadioAndControl:
    def test_radio_is_named_and_sorted(self, monkeypatch):
        payload = [
            {'driver_number': 4, 'date': '2026-10-11T12:01:00+00:00',
             'recording_url': 'https://example.test/b.mp3'},
            {'driver_number': 1, 'date': '2026-10-11T12:00:00+00:00',
             'recording_url': 'https://example.test/a.mp3'},
        ]
        fake, _ = _fake_client({'drivers': DRIVERS, 'team_radio': payload})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        frame = intervals.team_radio(11)

        assert list(frame.columns) == intervals.RADIO_COLUMNS
        assert frame['Driver'].tolist() == ['Max Verstappen', 'Lando Norris']
        assert frame['RecordingUrl'].iloc[0].endswith('a.mp3')

    def test_race_control_keeps_every_message(self, monkeypatch):
        payload = [
            {'driver_number': None, 'date': '2026-10-11T12:05:00+00:00',
             'lap_number': 12, 'category': 'SafetyCar', 'flag': 'SC',
             'message': 'SAFETY CAR DEPLOYED'},
            {'driver_number': 16, 'date': '2026-10-11T12:06:00+00:00',
             'lap_number': 13, 'category': 'CarEvent', 'flag': None,
             'message': 'CAR 16 UNDER INVESTIGATION'},
        ]
        fake, _ = _fake_client({'drivers': DRIVERS, 'race_control': payload})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        frame = intervals.race_control(11)

        assert list(frame.columns) == intervals.CONTROL_COLUMNS
        assert frame['Message'].tolist() == ['SAFETY CAR DEPLOYED',
                                            'CAR 16 UNDER INVESTIGATION']
        assert frame['Driver'].tolist() == ['', 'Charles Leclerc']
        assert frame['Lap'].tolist() == [12, 13]
        assert frame['Flag'].iloc[1] == ''


class TestFeedStatus:
    def test_no_session_key_says_so(self):
        status = intervals.feed_status(None)

        assert status['available'] is False
        assert 'OpenF1' in status['reason']
        assert status['intervals'] == 0

    def test_counts_every_feed(self, monkeypatch, gaps_payload):
        fake, _ = _fake_client({'drivers': DRIVERS, 'intervals': gaps_payload,
                                'team_radio': [], 'race_control': []})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        status = intervals.feed_status(11)

        assert status['available'] is True
        assert status['intervals'] == 5
        assert status['radio'] == 0
        assert status['control'] == 0

    def test_all_empty_feeds_explain_themselves(self, monkeypatch):
        fake, _ = _fake_client({})
        monkeypatch.setattr(openf1_client, 'cached_openf1', fake)

        status = intervals.feed_status(11)

        assert status['available'] is False
        assert 'no live gaps' in status['reason']


def test_number_handles_the_feed_spellings():
    assert intervals._number(None) is None
    assert intervals._number('') is None
    assert intervals._number('None') is None
    assert intervals._number('1.5') == 1.5
    assert intervals._number(3) == 3.0
    assert intervals._number(float('nan')) is None
    assert pd.isna(intervals._number(float('nan')))
