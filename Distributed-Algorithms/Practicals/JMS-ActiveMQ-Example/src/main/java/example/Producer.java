package example;

import org.apache.qpid.jms.JmsConnectionFactory;
import javax.jms.Connection;
import javax.jms.Session;
import javax.jms.Destination;
import javax.jms.DeliveryMode;
import javax.jms.TextMessage;
import javax.jms.MessageProducer;
import java.io.Console;

class Producer {

    public static void main(String[] args) throws Exception {

    	// Creation of a network connection to a specific JMS broker (e.g. ActiveMQ)
        JmsConnectionFactory factory = new JmsConnectionFactory("amqp://localhost:5672");
        Connection connection = factory.createConnection("admin", "password");
        connection.start();
        Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
        
        // A Destination is an address of a specific Topic or Queue hosted by the JMS broker.
        Destination destination = null;
        if(args.length > 0 && args[0].equalsIgnoreCase("Q")) {        	
        	destination = session.createQueue("Queue");	
        }else if(args.length > 0 && args[0].equalsIgnoreCase("T"))  {        	
        	destination = session.createTopic("Topic");        	
        }else {
        	System.out.println("Use Q for queue or T for topic");
        	connection.close();
        	System.exit(1);
        }
        
        // A MessageProducer can only send messages to one Topic or Queue. 
        MessageProducer producer = session.createProducer(destination);

        Console c = System.console();
        String response;
        do {        	
            response = c.readLine("Enter message (Press enter to send): ");
            TextMessage msg = session.createTextMessage(response);
            producer.send(msg);
        } while (!response.equalsIgnoreCase("CLOSE"));

        connection.close();
        System.exit(1);
    }
}