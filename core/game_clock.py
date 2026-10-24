"""Regulation basketball clocks. No browser access or model dependencies."""
import math
import re


def league_clock_settings(league_name=None):
    name = (league_name or 'NBA').upper()
    if any(key in name for key in ('WNBA', 'NBL', 'EURO', 'FIBA')):
        return 600, 4
    if 'NBA' in name:
        return 720, 4
    raise ValueError(f'Liga sin duracion configurada: {league_name}')


def normalize_event_clock(event, league_name=None):
    """Use the event's period, not the quotient of SofaScore's played field.

    In captured live responses, played's remainder matches the visible clock
    while its quotient disagrees with the visible period. Exact boundaries with
    a stopped clock are ambiguous unless the status explicitly identifies a break.
    No wall-clock extrapolation: timeouts and provider timestamps may be stale.
    """
    invalid = dict(clock_valid=False, clock_error='', played_seconds=0,
                   minutes_played=0., current_quarter=None,
                   current_quarter_display='Unknown', time_into_quarter=0,
                   remaining_in_quarter=0, total_remaining_seconds=0,
                   total_remaining_minutes=0., current_pace_per_minute=0.,
                   projected_final_score=0., current_total_score=0,
                   completion_percentage=0., is_live=False, is_finished=False,
                   is_overtime=False)
    try:
        timing = event.get('time') or {}
        status = event.get('status') or {}
        description = str(status.get('description', '')).lower()
        kind = status.get('type', 'unknown')
        finished = kind in ('finished', 'final')
        invalid.update(is_live=kind == 'inprogress', is_finished=finished)
        if ('overtime' in description or re.search(r'\bot\d*\b', description)
                or 'overtime' in str(event.get('lastPeriod', '')).lower()
                or any(v for side in ('homeScore', 'awayScore')
                       for k, v in (event.get(side) or {}).items()
                       if 'overtime' in k.lower())):
            invalid['is_overtime'] = True
            raise ValueError('Prorroga fuera del alcance del sistema')

        expected, periods = league_clock_settings(league_name) if league_name else (None, 4)
        supplied_length = timing.get('periodLength')
        if supplied_length is None and event.get('defaultPeriodLength') is not None:
            supplied_length = float(event['defaultPeriodLength']) * 60
        length = float(supplied_length if supplied_length is not None else expected or 720)
        count = float(timing.get('totalPeriodCount', event.get('defaultPeriodCount', periods)))
        if length not in (600, 720) or count != 4:
            raise ValueError('Formato de periodos no compatible')
        if expected is not None and length != expected:
            raise ValueError('La duracion del evento no coincide con la liga seleccionada')
        if event.get('defaultPeriodLength') is not None and float(event['defaultPeriodLength'])*60 != length:
            raise ValueError('Duraciones de cuarto contradictorias en el evento')
        duration = length * 4
        invalid.update(period_length=length, total_game_minutes=duration/60,
                       league_name=league_name)

        halftime = 'halftime' in description or 'half-time' in description or 'half time' in description
        match = re.search(r'([1-4])(?:st|nd|rd|th)?\s*(?:quarter|cuarto)', description)
        if not match:
            match = re.search(r'(?:quarter|cuarto|q)\s*([1-4])\b', description)
        quarter = int(match[1]) if match else None
        explicit = status.get('period')
        if explicit is not None:
            explicit = int(explicit)
            if explicit not in range(1, 5) or (quarter and explicit != quarter):
                raise ValueError('Periodo del evento contradictorio')
            quarter = explicit
        is_break = any(word in description for word in ('break', 'end of', 'ended', 'finished'))
        if quarter is None and is_break:
            last = re.fullmatch(r'period([1-4])', str(event.get('lastPeriod', '')))
            if last:
                quarter = int(last[1])
        if finished:
            quarter, elapsed, within = 4, duration, length
            stage = 'finished'
        elif halftime:
            quarter, elapsed, within = 2, 2*length, length
            stage = 'halftime'
        else:
            if quarter is None:
                # lastPeriod alone does not distinguish an active period from a break.
                raise ValueError('No se puede identificar el cuarto actual')
            if is_break:
                if quarter == 4:
                    raise ValueError('Fin del tiempo reglamentario: sin nueva prediccion')
                within = length
            else:
                raw = float(timing['played'])
                if not math.isfinite(raw) or raw < 0 or raw > duration:
                    raise ValueError('Reloj played no valido')
                within = raw % length
                if within == 0 and any(float((event.get(side) or {}).get(f'period{quarter}', 0) or 0) > 0
                                       for side in ('homeScore', 'awayScore')):
                    raise ValueError('Reloj al inicio del cuarto con puntos ya anotados en ese periodo')
                if within == 0 and raw > 0:
                    if raw != (quarter-1)*length or timing.get('clockRunning') is not True:
                        raise ValueError('Limite de cuarto ambiguo: sin estado de inicio o descanso fiable')
            elapsed = (quarter-1)*length + within
            stage = ('halftime' if quarter == 2 and is_break else
                     'q3_just_ended' if quarter == 3 and is_break else
                     'q4_progress' if quarter == 4 else
                     'q3_progress' if quarter == 3 else 'q1_or_q2')

        # A period cannot precede a later period in which points already exist.
        if any(float((event.get(side) or {}).get(f'period{q}', 0) or 0) > 0
               for side in ('homeScore', 'awayScore') for q in range(quarter+1, 5)):
            raise ValueError('El periodo no coincide con los parciales recibidos')
        score = sum(float((event.get(side) or {}).get('current', 0) or 0)
                    for side in ('homeScore', 'awayScore'))
        if not math.isfinite(score) or score < 0:
            raise ValueError('Marcador no valido')
        ppm = score / (elapsed/60) if elapsed else 0.
        result = dict(invalid, clock_valid=True, clock_error='',
                      clock_source='event_status_and_period_remainder',
                      raw_played_seconds=timing.get('played'),
                      played_seconds=elapsed, minutes_played=elapsed/60,
                      current_quarter=quarter, current_quarter_display=f'Q{quarter}',
                      quarter_stage=stage, time_into_quarter=within,
                      remaining_in_quarter=length-within,
                      total_remaining_seconds=duration-elapsed,
                      total_remaining_minutes=(duration-elapsed)/60,
                      current_total_score=score, current_pace_per_minute=ppm,
                      projected_final_score=ppm*(duration/60),
                      completion_percentage=elapsed/duration*100)
        return result
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError) as exc:
        invalid['clock_error'] = str(exc)
        return invalid
