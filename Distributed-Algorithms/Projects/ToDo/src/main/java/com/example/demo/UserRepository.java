package com.example.demo;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

public class UserRepository extends Repository {
	private Connection con = null;
	
	UserRepository(){
		super();
		con = Repository.getConnection();
	}
	
	public List<User> getAll(String userName, String password){
		List<User> users = new ArrayList<User>();
		Statement st;
		try {
			st = con.createStatement();
			ResultSet res = st.executeQuery("select * from users");
			while (res.next()) {
				User user = new User();
				user.setId((long)res.getInt("id"));
				user.setFirstName(res.getString("first_name"));
				user.setLastName(res.getString("last_name"));
				user.setUserName(res.getString("user_name"));
				user.setPassword(res.getString("password"));
				if(status(user.getId(), userName, password, con) != IsAuthorized) {
					user.setUserName(null);
					user.setPassword(null);
				}
				users.add(user);
			}
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return users;
	}
	
	public User getById(Long id, String userName, String password){
		List<User> users = getAll(userName, password);
		for(User user: users) {
			if(id == user.getId()) {
				return user;
			}
		}
		return new User();
	}
	
	public boolean add(User user) {
		try {
			Statement st = con.createStatement();
			st.executeUpdate("insert into users(first_name, last_name, user_name, password) " 
				+ "values ('" + user.getFirstName() + "', '" + user.getLastName() + "', '" 
				+ user.getUserName() + "', '" + createHash(user.getPassword()) + "')");
			return true;
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			System.out.println("Cannot add user!");
		}
		return false;
	}
	
	public User update(Long id, User user) {
		try {
			Statement st = con.createStatement();
			st.executeUpdate("update users "
					+ "set first_name='" + user.getFirstName() + "', "
						+ "last_name='" + user.getLastName() + "', "
						+ "user_name='" + user.getUserName() + "', "
						+ "password='" + user.getPassword() + "' "
					+ "where id=" + id);
			return user;
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			System.out.println("Cannot update user!");
		}
		return new User();
	}
	
	// Not Working! Should first delete all (userId, todoId) from other tables!
	public boolean remove(Long id, String userName, String password) {
		ResultSet res;
		String query = "delete from todo where ";
		boolean userHasProject = true;
		
		try {
			if(status(id, userName, password, con) != IsAuthorized) {
				throw new SQLException();
			}
			
			Statement st = con.createStatement();
			res = st.executeQuery("select * from user_todo where user_id=" + id);
			
			if(res.next()) {
				query += "id=" + res.getLong("todo_id");
			} else {
				userHasProject = false;
			}
			while(res.next()) {
				query += " or id=" + res.getLong("todo_id");
			}
			
			st.executeUpdate("delete from user_todo where user_id=" + id);
			st.executeUpdate("delete from users where id=" + id);
			if(userHasProject) {
				st.executeUpdate(query);
			}
			
			return true;
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			System.out.println("Cannot remove user!");
		}
		return false;
	}
}
