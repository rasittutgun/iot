import { Inject, Injectable, Injector, PLATFORM_ID } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';
import { environment } from '../../environments/environment.prod';

export interface IConfig {
  env: "production" | "development" | "test" | "local" | "stage";
}
//require('dotenv').config();

@Injectable({
  providedIn: 'root'
})
export class HttpService {
  evs:EventSource;
  private rfsUrl = environment.SERVER_HOST + '/rfs';
  private flushUrl = environment.SERVER_HOST + '/flush';
  private rfUpdates = environment.SERVER_HOST + '/getRfUpdates';

  private subj=new BehaviorSubject([]);


  constructor(private http: HttpClient) {}

  getRfs(): Observable<string> {
    return this.http.get(this.rfsUrl, { observe: 'body',responseType: 'text'});
  }


  returnAsObservable(): Observable<any> {
    return this.subj.asObservable();
  }

  getRfUpdates() {
    let subject=this.subj;

    if(typeof(EventSource) !== 'undefined') {
      this.evs=new EventSource(this.rfUpdates);

      this.evs.onopen=function(e) {
        console.log('Opening connection.Ready State is '+this.readyState);
      }

      this.evs.onmessage=function(e) {
        console.log('Message Received.Ready State is '+this.readyState);
        subject.next(e.data);
      }

      this.evs.addEventListener("timestamp",function(e) {
        console.log("Timestamp event Received.Ready State is "+this.readyState);
        subject.next(e["data"]);
      })

      this.evs.onerror=function(e) {
        console.log(e);
        if(this.readyState==0) {
          console.log('Reconnecting…');
        }
      }
    }
  }

  flushRedis(): Observable<string> {
    return this.http.get(this.flushUrl, { observe: 'body',responseType: 'text'});
  }

  stopExchangeUpdates() {
    this.evs.close();
  }
}
