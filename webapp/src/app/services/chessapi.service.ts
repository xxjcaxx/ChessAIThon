import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Scenario } from '../chess/scenario';
import { AuthSession, SupabaseClient, createClient } from '@supabase/supabase-js';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ChessapiService {
  private supabase: SupabaseClient;
  _session: AuthSession | null = null

  constructor(private http: HttpClient ){
    this.supabase = createClient(environment.supabaseUrl, environment.supabaseKey)
  }


  getRandomScenario(){
    return this.supabase
    .from('chessintion')
    .select(`id, fen, best, legal_moves_uci`)
    .eq('id', 100)
    .single()
  }


}
